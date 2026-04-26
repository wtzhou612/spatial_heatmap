import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label


base_dir   = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(base_dir, "data")
# 获取文件夹下所有的 csv 文件
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
# 按照文件名进行时间顺序排序
csv_files.sort()
df_list = []
current_frame_offset = 0

for file in csv_files:
    df_temp = pd.read_csv(file)
    if df_temp.empty:
        continue
        
    # 处理 frame_idx 不连续的问题（如果每个文件都从 0 开始记录）
    # 让当前文件的帧号加上之前的总帧数偏移量，保证全局 frame_idx 单调递增
    df_temp["frame_idx"] = df_temp["frame_idx"] + current_frame_offset
    current_frame_offset = df_temp["frame_idx"].max() + 1
    df_list.append(df_temp)

# 将所有读取到的 DataFrame 纵向拼接在一起，并重置内部索引
if df_list:
    df_raw = pd.concat(df_list, ignore_index=True)
else:
    raise FileNotFoundError("未在目标文件夹中找到有效的 CSV 数据文件！")
print(f"成功加载并拼接了 {len(csv_files)} 个文件，总数据行数: {len(df_raw)}")

# 提取数据，根据数据类型切割数据
NUM_TARGETS = 1
target_para = {}

FILTER_CONFIG = {
    "origin_radius"   : 0.10,
    "min_range"       : 0.30,
    "gap_fill_method" : "ffill",
    "max_gap_frames"  : 100000,
    "filled_type_val" : 0,
}

for i in range(NUM_TARGETS):
    suffix = f"_{i}"

    # Step 1：从原始数据中提取当前目标的列
    df_target = pd.DataFrame({
        "frame_idx" : df_raw["frame_idx"],
        "adl_rng"   : df_raw[f"adl_rng{suffix}"],
        "adl_vel"   : df_raw[f"adl_vel{suffix}"],
        "adl_agl"   : df_raw[f"adl_agl{suffix}"],
        "adl_type"  : df_raw[f"adl_type{suffix}"],
    })
    # Step 2：计算笛卡尔坐标（只做一次，不重复）
    agl_rad        = np.deg2rad(df_target["adl_agl"])
    df_target["x"] = df_target["adl_rng"] * np.sin(agl_rad)
    df_target["y"] = df_target["adl_rng"] * np.cos(agl_rad)
    # Step 3：数据清洗 —— 标记无效点为 NaN
    r_from_origin = np.sqrt(df_target["x"]**2 + df_target["y"]**2)
    invalid_mask  = (
        (r_from_origin        <= FILTER_CONFIG["origin_radius"]) |
        (df_target["adl_rng"] <  FILTER_CONFIG["min_range"])
    )
    df_target.loc[invalid_mask, ["x", "y", "adl_rng", "adl_agl", "adl_vel"]] = np.nan
    # Step 4：短空洞前向填充
    if FILTER_CONFIG["gap_fill_method"] == "ffill":
        nan_flag  = df_target["x"].isna()
        gap_group = (nan_flag != nan_flag.shift()).cumsum()
        gap_size  = nan_flag.groupby(gap_group).transform("sum")
        short_gap_mask = nan_flag & (gap_size <= FILTER_CONFIG["max_gap_frames"])
        long_gap_mask  = nan_flag & (gap_size >  FILTER_CONFIG["max_gap_frames"])
        cols_to_fill = ["x", "y", "adl_rng", "adl_agl", "adl_vel"]
        df_target[cols_to_fill] = df_target[cols_to_fill].ffill()
        df_target.loc[long_gap_mask, cols_to_fill] = np.nan
        df_target.loc[short_gap_mask, "adl_vel"]   = 0.0
        df_target.loc[short_gap_mask, "adl_type"]  = FILTER_CONFIG["filled_type_val"]
    # Step 5：丢弃仍为 NaN 的行
    n_raw     = len(df_target)
    df_target = df_target.dropna(subset=["x", "y"]).reset_index(drop=True)
    n_dropped = n_raw - len(df_target)
    print(f"Target {i+1}: {n_raw} rows → dropped {n_dropped} "
          f"({n_dropped/n_raw*100:.1f}%) → kept {len(df_target)}")
    # Step 6：存入字典
    target_para[i + 1] = df_target

# 雷达参数配置
RADAR_PARAMS = {
    "max_range"    : 6.5,          # 最大探测距离（米）
    "fov_half_deg" : 65,           # 单侧半视角（度）
    "frame_time"   : 0.128,        # 每帧时间（秒）
}

# ── 绘制雷达覆盖扇区 ──────────────────────────────────
def draw_radar_fov(ax, radar_params):
    R        = radar_params["max_range"]
    half_deg = radar_params["fov_half_deg"]

    # ── 扇区填充 ─────────────────────────────────────────────
    theta_start = 90 - half_deg
    theta_end   = 90 + half_deg
    sector = mpatches.Wedge(
        center=(0, 0), r=R,
        theta1=theta_start, theta2=theta_end,
        facecolor="steelblue", alpha=0.08,
        edgecolor="steelblue", linewidth=1.0,
        linestyle="--", zorder=0,
    )
    ax.add_patch(sector)

    # ── 在两侧绘制距离圆环和标签 ───────────────────────
    left_rad  = np.radians(90 + half_deg)   # 左侧边界角度
    right_rad = np.radians(90 - half_deg)   # 右侧边界角度

    for r in np.arange(2, R + 0.1, 2):
        # 绘制圆弧线
        angles = np.linspace(left_rad, right_rad, 200)
        ax.plot(r * np.cos(angles), r * np.sin(angles),
                color="steelblue", linewidth=0.5,
                linestyle=":", alpha=0.4, zorder=0)

        # 在圆弧左端添加标签
        ax.text(r * np.cos(left_rad) - 0.05,
                r * np.sin(left_rad),
                f"{r:.0f}m",
                fontsize=8, color="steelblue", alpha=0.75,
                ha="right", va="center")

        # 在圆弧右端添加标签
        ax.text(r * np.cos(right_rad) + 0.05,
                r * np.sin(right_rad),
                f"{r:.0f}m",
                fontsize=8, color="steelblue", alpha=0.75,
                ha="left", va="center")

    # ── 视场角(FOV)边界线和角度标签 ───────────────────────
    for sign, ha, angle_label in [
        (-1, "right", f"-{half_deg}°"),   # 左侧边界
        (+1, "left",  f"+{half_deg}°"),   # 右侧边界
    ]:
        angle_rad = np.radians(90 + sign * half_deg)
        ex = R * np.cos(angle_rad)
        ey = R * np.sin(angle_rad)

        # 绘制边界线
        ax.plot([0, ex], [0, ey],
                color="steelblue", linewidth=0.8,
                linestyle="--", alpha=0.5, zorder=0)

        # 沿着边界线在靠近雷达原点处添加角度标签
        label_r = 0.9   # 角度标签距离原点的距离（米）
        ax.text(label_r * np.cos(angle_rad),
                label_r * np.sin(angle_rad),
                angle_label,
                fontsize=8, color="steelblue", alpha=0.85,
                ha=ha, va="bottom",
                rotation=np.degrees(angle_rad) - 90)

    # ── 在原点附近绘制角度圆弧（视觉指示） ─────────────────
    arc_r   = 0.7
    arc_ang = np.linspace(left_rad, right_rad, 100)
    ax.plot(arc_r * np.cos(arc_ang), arc_r * np.sin(arc_ang),
            color="steelblue", linewidth=1.0, alpha=0.5, zorder=0)

    # 在小圆弧顶部添加总视场角标签
    ax.text(0, arc_r + 0.12,
            f"{half_deg * 2}° FOV",
            fontsize=8, color="steelblue", alpha=0.85,
            ha="center", va="bottom")

    # ── 雷达原点标记 ──────────────────────────────────────
    ax.plot(0, 0, "^", color="steelblue", markersize=8,
            markerfacecolor="white", markeredgewidth=1.5,
            zorder=5, label="Radar")


# # ── 热力图参数配置 ──────────────────────────────────────────────
HEATMAP_CONFIG = {
    # 网格分辨率
    "cart_grid_m"   : 0.2,     # 笛卡尔网格边长（米）
    "polar_r_bins"  : 26,       # 极坐标距离方向格数
    "polar_a_bins"  : 52,       # 极坐标角度方向格数

    # 平滑
    "gauss_sigma"   : 1.2,      # 高斯平滑核σ（格数），0 = 不平滑

    # 显示
    "cmap_all"      : "YlOrRd", # 综合热力图色彩
    "cmap_dynamic"  : "Blues",  # 动态点专用色彩
    "cmap_static"   : "Greens", # 静止点专用色彩
    "alpha_heatmap" : 0.8,     # 热力图层透明度
    "min_count"     : 2,        # 低于此计数的格子不着色（去噪）
    "figsize"       : (11, 9),

    # adl_type 中代表"动态"的值（根据你的数据定义调整）
    "dynamic_type_val" : 0,     # 0 = Dynamic，1 = Static
}

# ── 工具函数：汇总所有目标的 XY 坐标 ────────────────────────────
def collect_all_points(target_para, dynamic_only=False, static_only=False):
    xs_all, ys_all = [], []
    dyn_val = HEATMAP_CONFIG["dynamic_type_val"]

    for df in target_para.values():
        if dynamic_only:
            mask = df["adl_type"] == dyn_val
            xs_all.append(df.loc[mask, "x"].values)
            ys_all.append(df.loc[mask, "y"].values)
        elif static_only:
            mask = df["adl_type"] != dyn_val
            xs_all.append(df.loc[mask, "x"].values)
            ys_all.append(df.loc[mask, "y"].values)
        else:
            xs_all.append(df["x"].values)
            ys_all.append(df["y"].values)

    if not xs_all:
        return np.array([]), np.array([])
    return np.concatenate(xs_all), np.concatenate(ys_all)


# ══════════════════════════════════════════════════════════════════
#  家具检测模块：从热力图中提取家具轮廓
# ══════════════════════════════════════════════════════════════════

FURNITURE_CONFIG = {
    # 热区提取
    "heatmap_threshold"    : 0.15,
    "min_cell_count"       : 6,

    # 家具分类：adaptive 模式
    "density_mode"         : "adaptive",  # "adaptive" 或 "ratio"
    "density_ratio_thresh" : 1.0,         # 仅 mode="ratio" 时生效
    "large_area_thresh"    : 0.8,

    # 绘图
    "bed_color"            : "#E07B54",
    "chair_color"          : "#5B8DB8",
    "unknown_color"        : "#888888",
    "box_linewidth"        : 2.0,
    "box_alpha"            : 0.85,
    "label_fontsize"       : 9,
}


# ── 工具：从 target_para 提取分类热力图矩阵 ──────────────────────
def _build_split_heatmaps(target_para, x_edges, y_edges):
    """
    同时构建 All / Dynamic / Static 三张热力图矩阵。
    返回字典 {"all": H_all, "dynamic": H_dyn, "static": H_sta}
    """
    result = {}
    for mode in ["all", "dynamic", "static"]:
        xs, ys = collect_all_points(
            target_para,
            dynamic_only=(mode == "dynamic"),
            static_only=(mode == "static"),
        )
        if len(xs) == 0:
            H = np.zeros((len(y_edges)-1, len(x_edges)-1))
        else:
            H, _, _ = np.histogram2d(ys, xs, bins=[y_edges, x_edges])
        result[mode] = H.astype(float)
    return result


def _compute_adaptive_threshold(densities):
    if len(densities) <= 1:
        # 只有一个区域，无法比较，返回该值的 0.9 倍（该区域归为床）
        return densities[0] * 0.9, 0

    sorted_d = np.sort(densities)           # 升序排列
    gaps     = np.diff(sorted_d)            # 相邻差值
    split_idx = int(np.argmax(gaps))        # 间隔最大处的下标

    threshold = (sorted_d[split_idx] + sorted_d[split_idx + 1]) / 2.0

    print(f"  Adaptive density threshold: {threshold:.2f} hits/m²")
    print(f"  Density distribution: {np.round(sorted_d, 1).tolist()}")
    print(f"  Max gap: {gaps[split_idx]:.2f}  between index {split_idx} and {split_idx+1}")

    return threshold, split_idx


# ── 核心：检测热区并分类家具 ─────────────────────────────────────
def detect_furniture(target_para,
                     config=FURNITURE_CONFIG,
                     heatmap_cfg=HEATMAP_CONFIG,
                     radar_params=RADAR_PARAMS):

    R        = radar_params["max_range"]
    half_r   = np.radians(radar_params["fov_half_deg"])
    cell     = heatmap_cfg["cart_grid_m"]
    sigma    = heatmap_cfg["gauss_sigma"]

    x_lim   = R * np.sin(half_r)
    x_edges = np.arange(-x_lim, x_lim + cell, cell)
    y_edges = np.arange(0,       R    + cell,  cell)

    heatmaps = _build_split_heatmaps(target_para, x_edges, y_edges)
    H_all    = gaussian_filter(heatmaps["all"],    sigma=sigma) if sigma > 0 else heatmaps["all"]
    H_sta    = gaussian_filter(heatmaps["static"], sigma=sigma) if sigma > 0 else heatmaps["static"]

    cx = (x_edges[:-1] + x_edges[1:]) / 2
    cy = (y_edges[:-1] + y_edges[1:]) / 2
    CX, CY   = np.meshgrid(cx, cy)
    rng_grid = np.sqrt(CX**2 + CY**2)
    agl_grid = np.degrees(np.arctan2(CX, CY))
    fov_mask = (rng_grid <= R) & (np.abs(agl_grid) <= radar_params["fov_half_deg"])
    H_all[~fov_mask] = 0

    threshold          = H_all.max() * config["heatmap_threshold"]
    binary_mask        = (H_all >= threshold) & fov_mask
    labeled, n_regions = label(binary_mask)

    print(f"\n===== Furniture Detection =====")
    print(f"Threshold: {threshold:.2f}  |  Regions found: {n_regions}")

    # ── 第一次遍历：收集所有区域的原始特征 ────────────────────
    raw_regions = []

    for region_id in range(1, n_regions + 1):
        region_mask = (labeled == region_id)
        cell_count  = region_mask.sum()

        if cell_count < config["min_cell_count"]:
            continue

        rows, cols       = np.where(region_mask)
        row_min, row_max = rows.min(), rows.max()
        col_min, col_max = cols.min(), cols.max()

        x_min  = x_edges[col_min]
        x_max  = x_edges[col_max + 1]
        y_min  = y_edges[row_min]
        y_max  = y_edges[row_max + 1]
        width  = x_max - x_min
        height = y_max - y_min
        area   = cell_count * cell**2
        aspect = max(width, height) / max(min(width, height), 0.01)

        H_sta_region  = H_sta[region_mask].sum()
        H_all_region  = heatmaps["all"][region_mask].sum()
        static_ratio  = H_sta_region / max(H_all_region, 1)

        # 绝对密度（此时仅用于计算相对密度，不直接用于分类）
        hit_density = H_all_region / max(area, 0.01)

        raw_regions.append({
            "x_min"        : x_min,       "x_max"       : x_max,
            "y_min"        : y_min,       "y_max"       : y_max,
            "width_m"      : round(width,  2),
            "height_m"     : round(height, 2),
            "area_m2"      : round(area,   2),
            "aspect_ratio" : round(aspect, 2),
            "static_ratio" : round(static_ratio, 3),
            "hit_count"    : int(H_all_region),
            "hit_density"  : hit_density,   # 绝对密度，仅用于计算相对值
        })

    if not raw_regions:
        return [], x_edges, y_edges, H_all

    # ── 根据 mode 选择阈值计算方式 ────────────────────────────
    densities = np.array([r["hit_density"] for r in raw_regions])

    if config["density_mode"] == "adaptive":
        density_threshold, _ = _compute_adaptive_threshold(densities)
    else:
        # 原有 ratio 模式
        mean_density      = densities.mean()
        density_threshold = mean_density * config["density_ratio_thresh"]
        print(f"  Fixed ratio threshold: {density_threshold:.2f} hits/m²"
              f"  (mean={mean_density:.2f} × {config['density_ratio_thresh']})")

    # ── 第二次遍历：分类 ───────────────────────────────────────
    furniture_list = []

    for i, region in enumerate(raw_regions):
        is_bed = region["hit_density"] >= density_threshold

        furniture_label, color = _classify_furniture_adaptive(
            area        = region["area_m2"],
            is_bed      = is_bed,
            config      = config,
        )

        density_ratio = region["hit_density"] / max(densities.mean(), 0.01)

        furniture = {**region,
                     "label"         : furniture_label,
                     "density_ratio" : round(density_ratio, 2),
                     "is_bed"        : is_bed,
                     "color"         : color}
        furniture_list.append(furniture)

        print(f"  Region {i+1}: {furniture_label:14s} | "
              f"W={region['width_m']:.2f}m H={region['height_m']:.2f}m "
              f"Area={region['area_m2']:.2f}m² | "
              f"Density={region['hit_density']:.1f} hits/m²  "
              f"({'≥' if is_bed else '<'} threshold {density_threshold:.1f})")

    return furniture_list, x_edges, y_edges, H_all


def _classify_furniture_adaptive(area, is_bed, config):
    """
    分类逻辑：
      密度高于自适应阈值  →  Bed
      密度低于阈值且面积够大  →  Desk & Chair
      其他  →  Active Zone
    """
    if is_bed:
        return "Bed", config["bed_color"]
    elif area >= config["large_area_thresh"]:
        return "Desk & Chair", config["chair_color"]
    else:
        return "Active Zone", config["unknown_color"]


# ── 绘图：热力图 + 家具轮廓叠加 ─────────────────────────────────
def plot_heatmap_with_furniture(target_para,
                                config=FURNITURE_CONFIG,
                                heatmap_cfg=HEATMAP_CONFIG,
                                radar_params=RADAR_PARAMS):
    furniture_list, x_edges, y_edges, H_all = detect_furniture(
        target_para, config, heatmap_cfg, radar_params
    )

    if not furniture_list:
        print("No furniture regions detected. Try lowering 'heatmap_threshold' or 'min_cell_count'.")
        return

    # ── 绘图底层：热力图 ───────────────────────────────────────
    R      = radar_params["max_range"]
    half_r = np.radians(radar_params["fov_half_deg"])
    margin = 0.3

    fig, ax = plt.subplots(figsize=heatmap_cfg["figsize"])

    H_display = H_all.copy()
    H_display[H_display < heatmap_cfg["min_count"]] = np.nan

    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    im = ax.imshow(
        H_display,
        extent=extent,
        origin="lower",
        aspect="equal",
        cmap=heatmap_cfg["cmap_all"],
        alpha=heatmap_cfg["alpha_heatmap"],
        interpolation="bilinear",
        zorder=1,
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Hit Count (Smoothed)", fontsize=9)

    # ── 叠加家具轮廓 ───────────────────────────────────────────
    label_counter = {}   # 用于处理同类家具多个时编号

    for furn in furniture_list:
        lbl   = furn["label"]
        color = furn["color"]

        # 计数编号（同类家具第2个起加编号）
        label_counter[lbl] = label_counter.get(lbl, 0) + 1
        count = label_counter[lbl]
        display_label = f"{lbl} #{count}" if count > 1 else lbl

        # 矩形轮廓框
        rect = mpatches.FancyBboxPatch(
            (furn["x_min"], furn["y_min"]),
            furn["width_m"],
            furn["height_m"],
            boxstyle="round,pad=0.05",
            linewidth=config["box_linewidth"],
            edgecolor=color,
            facecolor=color,
            alpha=0.15,
            zorder=3,
        )
        ax.add_patch(rect)

        # 轮廓边框（单独画边线，控制不透明度）
        border = mpatches.FancyBboxPatch(
            (furn["x_min"], furn["y_min"]),
            furn["width_m"],
            furn["height_m"],
            boxstyle="round,pad=0.05",
            linewidth=config["box_linewidth"],
            edgecolor=color,
            facecolor="none",
            alpha=config["box_alpha"],
            zorder=4,
        )
        ax.add_patch(border)

        # 标签文字（框左上角）
        ax.text(
            furn["x_min"] + 0.05,
            furn["y_max"] - 0.05,
            f"{display_label}\n"
            f"{furn['width_m']:.1f}×{furn['height_m']:.1f}m",
            color=color,
            fontsize=config["label_fontsize"],
            fontweight="bold",
            va="top", ha="left",
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.2",
                      facecolor="white", alpha=0.75,
                      edgecolor=color, linewidth=0.8),
        )

    # ── 雷达 FOV ───────────────────────────────────────────────
    draw_radar_fov(ax, radar_params)

    # ── 图例 ───────────────────────────────────────────────────
    legend_patches = []
    seen_labels = set()
    for furn in furniture_list:
        if furn["label"] not in seen_labels:
            legend_patches.append(
                mpatches.Patch(color=furn["color"],
                               label=furn["label"],
                               alpha=0.8)
            )
            seen_labels.add(furn["label"])

    ax.legend(handles=legend_patches, loc="upper right",
              framealpha=0.85, fontsize=9)

    # ── 轴设置 ─────────────────────────────────────────────────
    x_max = R * np.sin(half_r) + margin
    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-margin, R + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title("Furniture Layout Detection from Radar Spatial Heatmap", fontsize=12, pad=12)
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ── 调用 ──────────────────────────────────────────────────────────
plot_heatmap_with_furniture(target_para)
