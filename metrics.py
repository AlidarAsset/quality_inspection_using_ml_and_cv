import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
import pandas as pd


def image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Image-level AUROC. labels: 0/1, scores: anomaly score per image."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, scores)


def pixel_auroc(masks: np.ndarray, score_maps: np.ndarray) -> float:
    """
    Pixel-level AUROC.
    masks:      [N, H, W]  binary ground truth
    score_maps: [N, H, W]  predicted anomaly scores
    """
    flat_gt    = masks.flatten().astype(int)
    flat_score = score_maps.flatten()
    if flat_gt.sum() == 0:
        return float("nan")
    return roc_auc_score(flat_gt, flat_score)


def per_region_overlap(masks: np.ndarray, score_maps: np.ndarray,
                        num_thresholds: int = 100) -> float:
    """
    PRO score (simplified).
    For each threshold, compute mean overlap per connected anomaly region.
    Returns area under the PRO-FPR curve (integrated 0..0.3).
    """
    from skimage.measure import label as sk_label

    thresholds = np.linspace(score_maps.min(), score_maps.max(), num_thresholds)
    pros, fprs = [], []

    for thresh in thresholds:
        pred_bin = (score_maps >= thresh)
        tp_pro_sum, region_count = 0.0, 0
        fp, tn = 0, 0

        for gt_mask, pred_mask in zip(masks, pred_bin):
            # True positive: overlap per region
            labeled = sk_label(gt_mask)
            for region_id in range(1, labeled.max() + 1):
                region = labeled == region_id
                if region.sum() == 0:
                    continue
                overlap = (pred_mask & region).sum() / region.sum()
                tp_pro_sum += overlap
                region_count += 1

            # False positive rate
            bg = ~gt_mask.astype(bool)
            fp += (pred_mask & bg).sum()
            tn += (~pred_mask & bg).sum()

        pros.append(tp_pro_sum / max(region_count, 1))
        fprs.append(fp / max(fp + tn, 1))

    pros = np.array(pros)
    fprs = np.array(fprs)

    # Integrate up to FPR=0.3
    mask_fpr = fprs <= 0.3
    if mask_fpr.sum() < 2:
        return float("nan")
    pro_score = auc(fprs[mask_fpr], pros[mask_fpr]) / 0.3
    return float(pro_score)


def optimal_f1(labels: np.ndarray, scores: np.ndarray):
    """Find threshold that maximises F1 score."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    return float(f1_scores[best_idx]), float(thresholds[best_idx])


def evaluate_model(model_name: str,
                   labels: np.ndarray,
                   image_scores: np.ndarray,
                   masks: np.ndarray,
                   score_maps: np.ndarray,
                   compute_pro: bool = True) -> dict:
    """
    Full evaluation. Returns dict of metrics.
    labels:       [N]      image-level binary labels
    image_scores: [N]      image-level anomaly scores
    masks:        [N,H,W]  pixel-level GT masks
    score_maps:   [N,H,W]  pixel-level anomaly maps
    """
    img_auc  = image_auroc(labels, image_scores)
    pix_auc  = pixel_auroc(masks, score_maps)
    f1, thr  = optimal_f1(labels, image_scores)

    result = {
        "model":          model_name,
        "image_auroc":    round(img_auc, 4),
        "pixel_auroc":    round(pix_auc, 4),
        "f1_score":       round(f1,      4),
        "threshold":      round(thr,     6),
    }

    if compute_pro:
        pro = per_region_overlap(masks, score_maps)
        result["pro_score"] = round(pro, 4)

    return result


def print_results(results: list[dict]):
    df = pd.DataFrame(results)
    
    print("  EVALUATION RESULTS")
    print(df.to_string(index=False))
    return df


def save_results(results: list[dict], path: str = "results/metrics.csv"):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    print(f"[Metrics] Saved to {path}")
    return df
