import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from matplotlib.lines import Line2D
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from utils import get_feature_set, genre_dict
from plot import plot_heat_map


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_data_with_pandas(csv_path):
    df = pd.read_csv(csv_path)
    # drop rows with NA
    df = df.dropna()
    return df


def load_data_with_utils(csv_path):
    # utils.get_feature_set returns (dataset, labelset) where labels are mapped to ints
    X_list, y_list = get_feature_set(csv_path)
    X = np.array(X_list)
    y = np.array(y_list)
    # filenames are not provided by utils.get_feature_set
    filenames = None
    feature_cols = [f'f{i}' for i in range(X.shape[1])] if X.ndim == 2 else None
    return X, y, filenames, feature_cols


def get_feature_label(df, label_col='label'):
    cols = list(df.columns)
    feature_cols = [c for c in cols if c not in [label_col, 'filename']]
    X = df[feature_cols].values
    y = df[label_col].values
    filenames = df['filename'].values if 'filename' in df.columns else None
    return X, y, filenames, feature_cols


def train_and_evaluate(X_train, X_test, y_train, y_test, fn_test, model_name, model, outdir, label_names=None, n_labels=None):
    # pipeline with scaling
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # probabilities if available
    y_proba = None
    if hasattr(pipe, 'predict_proba') or hasattr(pipe.named_steps['clf'], 'predict_proba'):
        try:
            y_proba = pipe.predict_proba(X_test)
        except Exception:
            y_proba = None

    acc = metrics.accuracy_score(y_test, y_pred)
    report = metrics.classification_report(y_test, y_pred, output_dict=False)
    # use consistent label ordering for confusion matrix when n_labels provided
    if n_labels is not None:
        labels_for_cm = list(range(n_labels))
    else:
        labels_for_cm = np.unique(y_test)
    cm = metrics.confusion_matrix(y_test, y_pred, labels=labels_for_cm)

    # save predictions
    preds_df = pd.DataFrame({'filename': fn_test if fn_test is not None else np.arange(len(y_test)),
                             'y_true': y_test,
                             'y_pred': y_pred})
    if y_proba is not None:
        # save max probability
        preds_df['y_proba_max'] = y_proba.max(axis=1)
    preds_csv = os.path.join(outdir, f'predictions_{model_name}.csv')
    preds_df.to_csv(preds_csv, index=False)

    # save classification report
    with open(os.path.join(outdir, f'classification_report_{model_name}.txt'), 'w') as f:
        f.write(f'Accuracy: {acc}\n\n')
        f.write(str(report))

    # plot confusion matrix using shared plot helper (accepts optional label names)
    try:
        # if label_names provided, order ticks accordingly
        plot_heat_map(y_test, y_pred, outdir, f'confusion_{model_name}', show=False, labels=label_names)
    except Exception:
        # fallback: simple heatmap without names
        labels = np.unique(np.concatenate([y_test, y_pred]))
        labels = [str(l) for l in labels]
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics.confusion_matrix(y_test, y_pred, labels=np.array(labels, dtype=object).astype(object)),
                    xticklabels=labels, yticklabels=labels, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name} (acc={acc:.3f})')
        plt.ylabel('True')
        plt.xlabel('Pred')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'confusion_{model_name}.png'))
        plt.close()

    return {'model': model_name, 'accuracy': acc, 'report': report, 'preds_csv': preds_csv, 'cm': cm}


def plot_pca(X, y_true, y_pred, labels, outpath_prefix, n_components=2):
    """Plot PCA scatter in 2D or 3D.

    Args:
        X: feature matrix
        y_true: true labels (ints or strs)
        y_pred: predicted labels (ints or strs)
        labels: list of label names (indexed by int)
        outpath_prefix: prefix for saved files
        n_components: 2 or 3
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    Xp = pca.fit_transform(Xs)

    # helper to get display names for a label array
    def disp_names(arr):
        res = []
        for i in arr:
            if isinstance(i, (int, np.integer)):
                if labels is not None and int(i) < len(labels):
                    res.append(labels[int(i)])
                else:
                    res.append(str(i))
            else:
                res.append(str(i))
        return res

    # Colors: use tab10 up to number of labels
    n_labels = len(labels) if labels is not None else len(np.unique(np.concatenate([y_true, y_pred])))
    palette = sns.color_palette('tab10', n_colors=max(3, n_labels))

    if n_components == 2:
        # True label scatter (2D)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=Xp[:, 0], y=Xp[:, 1], hue=disp_names(y_true), palette=palette, s=30, alpha=0.8)
        plt.title('PCA (2D) - True labels')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(outpath_prefix + '_pca_true.png', bbox_inches='tight')
        plt.close()

        # Pred label scatter (2D)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=Xp[:, 0], y=Xp[:, 1], hue=disp_names(y_pred), palette=palette, s=30, alpha=0.8)
        plt.title('PCA (2D) - Predicted labels')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(outpath_prefix + '_pca_pred.png', bbox_inches='tight')
        plt.close()
    elif n_components == 3:
        # 3D scatter: create figure and 3D axes
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
        # map labels to colors
        unique_labels = list(dict.fromkeys(disp_names(y_true)))
        label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
        colors = [palette[label_to_idx[lab] % len(palette)] for lab in disp_names(y_true)]
        ax.scatter(Xp[:, 0], Xp[:, 1], Xp[:, 2], c=colors, s=25, depthshade=True, alpha=0.9)
        # construct legend handles
        handles = []
        for i, lab in enumerate(unique_labels):
            handles.append(Line2D([0], [0], marker='o', color='w', label=lab,
                                    markerfacecolor=palette[i % len(palette)], markersize=6))
        ax.set_title('PCA (3D) - True labels')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(outpath_prefix + '_pca_3d_true.png', bbox_inches='tight')
        plt.close()

        # Predicted labels (3D)
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
        unique_labels_pred = list(dict.fromkeys(disp_names(y_pred)))
        label_to_idx_pred = {lab: i for i, lab in enumerate(unique_labels_pred)}
        colors = [palette[label_to_idx_pred[lab] % len(palette)] for lab in disp_names(y_pred)]
        ax.scatter(Xp[:, 0], Xp[:, 1], Xp[:, 2], c=colors, s=25, depthshade=True, alpha=0.9)
        handles = []
        for i, lab in enumerate(unique_labels_pred):
            handles.append(Line2D([0], [0], marker='o', color='w', label=lab,
                                    markerfacecolor=palette[i % len(palette)], markersize=6))
        ax.set_title('PCA (3D) - Predicted labels')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(outpath_prefix + '_pca_3d_pred.png', bbox_inches='tight')
        plt.close()
    else:
        raise ValueError('n_components must be 2 or 3')


def main():
    parser = argparse.ArgumentParser(description='Traditional ML classifiers (SVM, KNN) on feature CSV')
    parser.add_argument('--csv', type=str, default='Data/features_30_sec.csv', help='features CSV file')
    parser.add_argument('--label', type=str, default='label', help='name of label column')
    parser.add_argument('--outdir', type=str, default='result/traditional_ml', help='output directory')
    parser.add_argument('--test-size', type=float, default=0.2, help='test split fraction')
    parser.add_argument('--kfold', type=int, default=0, help='number of folds for Stratified K-Fold CV (0 = disabled)')
    parser.add_argument('--random-state', type=int, default=42, help='random state for splits and shuffling')
    args = parser.parse_args()

    ensure_dir(args.outdir)

    # Prefer utils.get_feature_set for feature CSVs to reuse existing mapping (labels -> ints)
    try:
        X, y, filenames, feature_cols = load_data_with_utils(args.csv)
        used_utils = True
    except Exception:
        # fallback to pandas
        df = load_data_with_pandas(args.csv)
        X, y, filenames, feature_cols = get_feature_label(df, label_col=args.label)
        used_utils = False

    # keep filenames aligned for saving predictions
    fn_array = filenames if filenames is not None else np.arange(len(y))

    # build label name mapping for visualization
    if used_utils:
        # genre_dict maps name->int; build reverse mapping ordered by int
        rev_map = {v: k for k, v in genre_dict.items()}
        # ensure labels list ordered by index
        max_label = int(np.max(y))
        label_names = [rev_map.get(i, str(i)) for i in range(max_label + 1)]
    else:
        # use LabelEncoder to get ordered class names
        le = LabelEncoder()
        y = le.fit_transform(y)
        label_names = list(le.classes_)

    results = []

    classifiers = {
        'SVM': SVC(kernel='rbf', probability=True, random_state=args.random_state),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }

    if args.kfold and args.kfold > 1:
        # use Stratified K-Fold on the full dataset
        skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.random_state)
        agg_results = {name: [] for name in classifiers.keys()}
        # prepare aggregated confusion matrices
        # determine number of labels
        n_labels = len(label_names) if label_names is not None else len(np.unique(y))
        agg_cm = {name: np.zeros((n_labels, n_labels), dtype=int) for name in classifiers.keys()}
        fold_idx = 0
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f'K-Fold: training fold {fold_idx + 1}/{args.kfold} ...')
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            fn_test = fn_array[test_idx] if fn_array is not None else None
            for name, clf in classifiers.items():
                model_name = f"{name}_fold{fold_idx+1}"
                out = train_and_evaluate(X_train, X_test, y_train, y_test, fn_test, model_name, clf, args.outdir, label_names=label_names, n_labels=n_labels)
                agg_results[name].append(out['accuracy'])
                results.append(out)
                # accumulate confusion matrix
                try:
                    agg_cm[name] += out.get('cm', np.zeros((n_labels, n_labels), dtype=int))
                except Exception:
                    # in case shapes mismatch, skip accumulation for this fold
                    pass

        # summarize k-fold results per model
        summary_rows = []
        for name, accs in agg_results.items():
            mean_acc = float(np.mean(accs))
            std_acc = float(np.std(accs))
            summary_rows.append({'model': name, 'kfold_mean_accuracy': mean_acc, 'kfold_std_accuracy': std_acc})
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(args.outdir, 'kfold_metrics_summary.csv'), index=False)
        # save aggregated confusion matrices
        for name, cm_sum in agg_cm.items():
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_sum, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
            plt.title(f'Aggregated Confusion Matrix - {name} (k={args.kfold})')
            plt.ylabel('True')
            plt.xlabel('Pred')
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f'confusion_{name}_kfold_agg.png'))
            plt.close()
    else:
        # single train/test split
        X_train, X_test, y_train, y_test, fn_train, fn_test = train_test_split(
            X, y, fn_array, test_size=args.test_size, random_state=args.random_state, stratify=y)

        for name, clf in classifiers.items():
            print(f'Training {name} ...')
            out = train_and_evaluate(X_train, X_test, y_train, y_test, fn_test, name, clf, args.outdir, label_names=label_names, n_labels=len(label_names) if label_names is not None else None)
            results.append(out)

    # PCA visualization on the whole dataset (true labels)
    # produce both 2D and 3D PCA plots
    plot_pca(X, y, y, label_names, os.path.join(args.outdir, 'all'), n_components=2)
    try:
        plot_pca(X, y, y, label_names, os.path.join(args.outdir, 'all'), n_components=3)
    except Exception as e:
        print('3D PCA plotting failed (falling back to 2D only):', e)

    # Save summary metrics
    summary = pd.DataFrame([{ 'model': r['model'], 'accuracy': r['accuracy'], 'preds_csv': r['preds_csv']} for r in results])
    summary.to_csv(os.path.join(args.outdir, 'metrics_summary.csv'), index=False)

    print('Done. Results and plots saved to', args.outdir)


if __name__ == '__main__':
    main()
