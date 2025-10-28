import os
import time
import pickle
from typing import List, Tuple
import gc
import numpy as np
import faiss
from tqdm import tqdm

INPUT_VECTORS_PATH = 'data/reduced_vectors_pca_128d_norm.npy'
OUTPUT_RESULTS_PATH = 'data/pca_128_hd_agc_results.pkl'

HNSW_M = 64
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 100
K_NEIGHBORS = 32


S_MIN = 0.83   
MIN_PTS = 3  
MAX_CLUSTER_SIZE = 6   


REFINEMENT_ITERATIONS = 5    
S_MERGE = 0.80               
S_MIN_FOR_OUTLIER = 0.78     
S_MIN_FOR_REFINEMENT = 0.78  


def to_float32(x: np.ndarray) -> np.ndarray:
    if x.dtype != np.float32:
        return x.astype('float32')
    return x


def build_hnsw_graph(vectors: np.ndarray, m: int, ef_construction: int, ef_search: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
    n, d = vectors.shape
    print(f"--- 阶段一: 构建HNSW图 (M={m}, efC={ef_construction}) ---")

    vectors = to_float32(vectors)

    index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction

    print("正在构建索引...")
    t0 = time.time()
    index.add(vectors)
    t1 = time.time()
    print(f"索引构建完成，耗时: {t1 - t0:.2f}s")

    raw_k = k + 1
    print(f"--- 正在搜索 {k} 个近邻 (efS={ef_search}) ---")
    index.hnsw.efSearch = max(ef_search, raw_k)

    t0 = time.time()
    sims_raw, inds_raw = index.search(vectors, raw_k)
    t1 = time.time()
    print(f"近邻搜索完成，耗时: {t1 - t0:.2f}s")

    final_inds = np.empty((n, k), dtype=np.int64)
    final_sims = np.empty((n, k), dtype=np.float32)

    for i in range(n):
        mask = (inds_raw[i] != i)
        filtered_inds = inds_raw[i, mask]
        filtered_sims = sims_raw[i, mask]
        
        cnt = min(k, len(filtered_inds))
        final_inds[i, :cnt] = filtered_inds[:cnt]
        final_sims[i, :cnt] = filtered_sims[:cnt]
        
        if cnt < k:
            final_inds[i, cnt:] = -1
            final_sims[i, cnt:] = -np.inf

    return final_inds, final_sims


def run_core_clustering(vectors: np.ndarray, neighbor_indices: np.ndarray,
                        neighbor_similarities: np.ndarray, s_min: float,
                        min_pts: int, max_size: int) -> Tuple[List[List[int]], List[int]]:
    n = vectors.shape[0]
    print("\n--- 阶段二: 运行DTCC核心聚类 ---")

    qualified_neighbors_count = np.sum(neighbor_similarities >= s_min, axis=1)
    is_core = qualified_neighbors_count >= min_pts
    print(f"识别出 {np.sum(is_core)} 个核心点。")

    visited_mask = np.zeros(n, dtype=bool)
    proto_clusters: List[List[int]] = []
    
    for i in tqdm(range(n), desc="寻找聚类种子"):
        if visited_mask[i] or not is_core[i]:
            continue

        current_cluster = []
        queue = [i]
        
        points_in_this_bfs = {i}

        head = 0
        while head < len(queue) and len(current_cluster) < max_size:
            p_idx = queue[head]
            head += 1
            current_cluster.append(p_idx)
            
            if is_core[p_idx]:
                for neighbor_idx, sim in zip(neighbor_indices[p_idx], neighbor_similarities[p_idx]):
                    neighbor_idx = int(neighbor_idx)
                    if neighbor_idx != -1 and sim >= s_min and not visited_mask[neighbor_idx] and neighbor_idx not in points_in_this_bfs:
                        points_in_this_bfs.add(neighbor_idx)
                        queue.append(neighbor_idx)
        
        for point_id in current_cluster:
            visited_mask[point_id] = True
            
        proto_clusters.append(current_cluster)
        
    outliers = np.where(~visited_mask)[0].tolist()
    
    total_points_in_clusters = sum(len(c) for c in proto_clusters)
    total_accounted_for = total_points_in_clusters + len(outliers)
    
    assert total_accounted_for == n, f"点总数不匹配! 聚类点数: {total_points_in_clusters}, 离群点数: {len(outliers)}, 总计: {total_accounted_for}, 期望: {n}"

    return proto_clusters, outliers

def calculate_centroids(vectors: np.ndarray, clusters: List[List[int]]) -> np.ndarray:
    d = vectors.shape[1]
    centroid_list = []
    for c in clusters:
        if c:
            cent = np.mean(vectors[c], axis=0)
            norm = np.linalg.norm(cent)
            if norm > 0:
                cent /= norm
            centroid_list.append(cent)
    if not centroid_list:
        return np.zeros((0, d), dtype=np.float32)
    return np.array(centroid_list, dtype=np.float32)

def merge_clusters_dsu(vectors: np.ndarray, clusters: List[List[int]], s_merge: float, max_size: int) -> List[List[int]]:
    num_clusters = len(clusters)
    if num_clusters <= 1: return clusters

    parent = list(range(num_clusters))
    sizes = [len(c) for c in clusters]
    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            if sizes[root_i] < sizes[root_j]: root_i, root_j = root_j, root_i
            parent[root_j] = root_i
            sizes[root_i] += sizes[root_j]

    centroids = calculate_centroids(vectors, clusters)
    if centroids.shape[0] == 0: return clusters

    sim_mat = centroids @ centroids.T
    pairs = np.argwhere(np.triu(sim_mat > s_merge, k=1))
    
    sorted_pairs = sorted(pairs, key=lambda p: sim_mat[p[0], p[1]], reverse=True)

    for i, j in sorted_pairs:
        root_i, root_j = find(i), find(j)
        if root_i != root_j and sizes[root_i] + sizes[root_j] <= max_size:
            union(i, j)

    merged = {}
    for idx in range(num_clusters):
        root = find(idx)
        if root not in merged: merged[root] = []
        merged[root].extend(clusters[idx])
    
    return list(merged.values())

def run_global_refinement(vectors: np.ndarray, proto_clusters: List[List[int]], outliers: List[int],
                          iterations: int, max_size: int, s_min_for_outlier: float, 
                          s_min_for_refinement: float, s_merge: float) -> Tuple[List[List[int]], List[int]]:
    if not proto_clusters and not outliers:
        return [], []
    print(f"\n--- 阶段三: 全局优化与精炼 (迭代次数={iterations}) ---")

    refined_clusters = [list(c) for c in proto_clusters]
    current_outliers = set(outliers)

    for it in range(iterations):
        print(f"\n--- 迭代 {it+1}/{iterations} ---")
        if not refined_clusters: break
        
        iteration_start_outliers_count = len(current_outliers)
        print(f"迭代开始时，离群点数量: {iteration_start_outliers_count}")

        if current_outliers:
            centroids = calculate_centroids(vectors, refined_clusters)
            
            if centroids.shape[0] > 0:
                outlier_indices = np.array(list(current_outliers), dtype=np.int64)
                
                sims = vectors[outlier_indices] @ centroids.T
                
                best_idx = np.argmax(sims, axis=1)
                best_sims = sims[np.arange(len(sims)), best_idx]
                
                order = np.argsort(best_sims)[::-1]
                
                assigned_points = set()
                for pos in order:
                    if best_sims[pos] < s_min_for_outlier: break
                    
                    point_to_assign = outlier_indices[pos]
                    tgt_cluster_idx = best_idx[pos]

                    if len(refined_clusters[tgt_cluster_idx]) < max_size:
                        refined_clusters[tgt_cluster_idx].append(point_to_assign)
                        assigned_points.add(point_to_assign)
                
                num_assigned = len(assigned_points)
                if num_assigned > 0:
                    current_outliers.difference_update(assigned_points)
                print(f"成功招安 {num_assigned} 个离群点。")

                del sims, best_idx, best_sims, order, outlier_indices, assigned_points
                gc.collect()

        print("正在纯化现有簇...")
        newly_expelled_outliers = []
        temp_clusters = []
        centroids = calculate_centroids(vectors, refined_clusters)
        if centroids.shape[0] > 0:
            for i, c in enumerate(refined_clusters):
                if len(c) > 1:
                    cluster_points_indices = np.array(c)
                    sims_to_centroid = vectors[cluster_points_indices] @ centroids[i]
                    keep_mask = sims_to_centroid >= s_min_for_refinement
                    
                    kept_points = cluster_points_indices[keep_mask].tolist()
                    if kept_points:
                        temp_clusters.append(kept_points)

                    expelled_points = cluster_points_indices[~keep_mask].tolist()
                    if expelled_points:
                        newly_expelled_outliers.extend(expelled_points)
                elif c:
                    temp_clusters.append(c)

            refined_clusters = temp_clusters
            if newly_expelled_outliers:
                current_outliers.update(newly_expelled_outliers)
            print(f"从簇中开除 {len(newly_expelled_outliers)} 个点变为离群点。")
            
            del centroids, temp_clusters, newly_expelled_outliers
            gc.collect()

        print(f"迭代结束时，离群点数量: {len(current_outliers)}")


    print("\n--- 正在进行最终的簇合并 ---")
    final_clusters = merge_clusters_dsu(vectors, refined_clusters, s_merge, max_size)
    print(f"合并完成，簇数量从 {len(refined_clusters)} 减少到 {len(final_clusters)}。")

    clustered_points = set()
    for c in final_clusters:
        clustered_points.update(c)
    
    all_points = set(range(vectors.shape[0]))
    final_outliers = sorted(list(all_points - clustered_points))

    return final_clusters, final_outliers

def main():
    print("--- HD-AGC v6.0 最终版 启动 ---")
    if not os.path.exists(INPUT_VECTORS_PATH):
        raise FileNotFoundError(f"输入文件不存在: {INPUT_VECTORS_PATH}")

    vectors = np.load(INPUT_VECTORS_PATH)
    vectors = to_float32(vectors)
    n, d = vectors.shape
    print(f"向量加载完成, n={n}, d={d}")

    neighbor_indices, neighbor_sims = build_hnsw_graph(vectors, HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH, K_NEIGHBORS)

    proto_clusters, outliers = run_core_clustering(vectors, neighbor_indices, neighbor_sims, S_MIN, MIN_PTS, MAX_CLUSTER_SIZE)
    print(f"阶段二完成，形成 {len(proto_clusters)} 个原生簇和 {len(outliers)} 个离群点。")

    final_clusters, final_outliers = run_global_refinement(vectors, proto_clusters, outliers, REFINEMENT_ITERATIONS,
                                                           MAX_CLUSTER_SIZE, S_MIN_FOR_OUTLIER, S_MIN_FOR_REFINEMENT, S_MERGE)

    clustered_points = set()
    for c in final_clusters:
        clustered_points.update(c)
    all_points = set(range(n))
    final_outliers = sorted(list(all_points - clustered_points))

    results = { 'clusters': final_clusters, 'outliers': final_outliers, 'params': { 'S_min': S_MIN, 'MinPts': MIN_PTS, 'k': K_NEIGHBORS, 'HNSW_M': HNSW_M, 'efC': HNSW_EF_CONSTRUCTION, 'efS': HNSW_EF_SEARCH, 'S_merge': S_MERGE, 'S_min_outlier': S_MIN_FOR_OUTLIER, 'S_min_refine': S_MIN_FOR_REFINEMENT } }

    out_dir = os.path.dirname(OUTPUT_RESULTS_PATH)
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    with open(OUTPUT_RESULTS_PATH, 'wb') as f: pickle.dump(results, f)

    print(f"结果已成功保存到: {OUTPUT_RESULTS_PATH}")
    
    num_clustered = sum(len(c) for c in final_clusters)
    total_points = num_clustered + len(final_outliers)
    print(f"\n最终统计: {len(final_clusters)} 个簇, {len(final_outliers)} 个离群点, 共 {total_points} 个点。")
    assert total_points == n, "最终点数与初始点数不匹配！"

if __name__ == '__main__':
    main()