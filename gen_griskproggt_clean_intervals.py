import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm

# Constants to activate exclusion rules
QUALITY_EXCL = -2
EYESIDE_EXCL = 0
SHAPE_EXCL = 0
DISCD_DETECT_EXCL = 0
DISCHU_EXCL = 0

# Constants to control the interval definition
MAX_DIFF = 3000
M = 0
N = 5
O = 3
INTERVAL = 1

plt.switch_backend('Agg')

def get_case(f):
    return f.replace("\\", "/").split('/')[-1].split('_')[0][:-1]

def get_eye(f):
    return f.replace("\\", "/").split('/')[-1].split('_')[0][-1]

def get_pvalue(regr, params, X, y, tail='left'):
    predictions = regr.predict(X)
    r2 = r2_score(y, predictions)
    newX = pd.DataFrame({'Constant': np.ones(len(X))}).join(pd.DataFrame(X))
    MSE = (sum((y - predictions)**2)) / (len(newX) - len(newX.columns))
    var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    sd_b = np.sqrt(var_b)
    # Prevent divide by zero error
    with np.errstate(divide='ignore', invalid='ignore'):
        ts_b = np.where(sd_b != 0, params / sd_b, 0)
    if tail == 'right':
        pval = t.cdf(ts_b, len(newX) - 2)
    elif tail == 'left':
        pval = 1 - t.cdf(ts_b, len(newX) - 2)
    elif tail == 'both':
        pval = 2 * (1 - t.cdf(abs(ts_b), len(newX) - 2))
    return pval, r2

def reg_ols(X, y, tail='left'):
    X = np.array(X).reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    params = np.append(reg.intercept_, reg.coef_)
    pval, r2 = get_pvalue(reg, params, X, y, tail)
    return pd.Series({'slope': reg.coef_.item(), 'pval': pval[1], 'intercept': reg.intercept_.item(), 'r2': r2})

def rem_exclude(df, quality_constant, conditions_dict):
    conditions = [df['quality'] <= quality_constant]
    for col, (constant, activate) in conditions_dict.items():
        if activate:
            conditions.append(df[col] == constant)
    return df[np.logical_and.reduce(conditions)]

CONDITIONS = {
    'eyeside_exclude': (0, EYESIDE_EXCL),
    'Shape_exclude': (0, SHAPE_EXCL),
    'DiscDetect_exclude': (0, DISCD_DETECT_EXCL),
    'DiscHu_exclude': (0, DISCHU_EXCL),
}

def get_dfexcl(df, clean):
    merged = df.merge(clean, how='left', indicator=True)
    df = df[merged['_merge'] == 'left_only']
    return df

def extract_intervals(dates, m, n, o, maxdiff=550):
    def is_valid_interval(interval):
        start_date = interval[0]
        end_date = interval[-1]
        return (end_date - start_date) // 365 >= m and (end_date - start_date) // 365 < n

    intervals = {}
    for i in range(len(dates)):
        current_interval = [dates[i]]
        for j in range(i + 1, len(dates)):
            if dates[j] - current_interval[-1] <= maxdiff:
                current_interval.append(dates[j])
            else:
                break
            if len(current_interval) >= o and is_valid_interval(current_interval):
                start_date = current_interval[0]
                if start_date in intervals:
                    if len(current_interval) > len(intervals[start_date]):
                        intervals[start_date] = current_interval.copy()
                else:
                    intervals[start_date] = current_interval.copy()
    return list(intervals.values())

def generate_line_colors(num_intervals):
    cmap = plt.get_cmap('tab10')
    return [cmap(i) for i in np.linspace(0, 1, num_intervals)]

def generate_output_series(ols0, ols2, df_clean, df_excl, examyears_clean, case, eye):
    return pd.Series(data={
        'case': case,
        'eye': eye,
        'sex': df_clean.Sex.iloc[0],
        'age': df_clean.Age.iloc[0],
        'baseline_md': df_clean.MD.iloc[0],
        'baseline_grisk_reg0': df_clean['G-RISK_reg0'].iloc[0],
        'grisk_slope_reg0': ols0.slope,
        'grisk_pval_reg0': ols0.pval,
        'r2_reg0': ols0.r2,
        'md_slope': ols2.slope,
        'md_pval': ols2.pval,
        'visits': len(df_clean),
        'FU_years': examyears_clean[-1] - examyears_clean[0],
        'img': df_clean['ImgName'].iloc[0], 
        'removal%': len(df_excl) / (len(df_excl) + len(df_clean)),
        'mean_hu': df_clean['DiscHu'].mean(),
        'mean_dr': df_clean['DiscRatio'].mean(),
        'std_quality': df_clean['quality'].std(),
        'std_eyeside': df_clean['eyeside'].std(),
        'timebetwvis': df_clean.timebetwvis.dropna().mean()
    })

def grisk_progplot(case, eye):
    plt.rcParams['font.family'] = 'Arial'

    df = pd.read_csv(f'./progplots/{case}/{case}{eye}_ODinfo.csv')

    df.timebetwvis.replace(0, pd.NA, inplace=True)
    df.timebetwvis.ffill(inplace=True)

    df_clean = rem_exclude(df, quality_constant=QUALITY_EXCL, conditions_dict=CONDITIONS)

    if 'MD' in df_clean.columns:
        df_MD = df_clean.dropna(subset=['MD'])
    else:
        df_clean['MD'] = pd.NA
        df_MD = pd.DataFrame()

    try:
        assert len(df_clean) > 2, "Less than three visits in DataFrame"
    except AssertionError:
        return

    df_excl = get_dfexcl(df, df_clean)

    examdates_all = df.Examdate.tolist()
    x = np.linspace(0, max(examdates_all), len(examdates_all))

    examdates_clean = df_clean.Examdate.tolist()
    examdates_excl = df_excl.Examdate.tolist()

    if not df_MD.empty:
        examdates_MD = df_MD.Examdate.tolist()

    ols0 = reg_ols(examdates_clean, df_clean['G-RISK_reg0'])
    coef_ols0 = ols0['slope'] * x + ols0['intercept']

    if not df_MD.empty and len(df_MD) > 2:
        ols1 = reg_ols(examdates_MD, df_MD['MD'], tail='right')
        coef_ols1 = ols1['slope'] * x + ols1['intercept']
    else:
        ols1 = None

    fig, ax1 = plt.subplots(1, 1, figsize=(4, 4), facecolor='#F8F5FF', dpi=150)
    ax1.scatter(examdates_clean, df_clean['G-RISK_reg0'], c='none', edgecolors='b')
    ax1.scatter(examdates_excl, df_excl['G-RISK_reg0'], c='none', edgecolors='red')

    output = []

    if INTERVAL:
        intervals = extract_intervals(examdates_clean, m=M, n=N, o=O, maxdiff=MAX_DIFF)
        line_colors = generate_line_colors(len(intervals))

        for i, interval in enumerate(intervals):
            df_clean_int = df_clean[df_clean.Examdate.isin(interval)]
            df_excl_int = df_excl[df_excl.Examdate.isin(interval)]
            if not df_MD.empty:
                df_MD_int = df_MD[df_MD.Examdate.isin(interval)]
            else:
                df_MD_int = [pd.NA]

            ols0_int = reg_ols(interval, df_clean_int['G-RISK_reg0'])
            y_pred_interval = [ols0_int.intercept + ols0_int.slope * x for x in interval]

            if not df_MD.empty and len(df_MD_int) > 2:
                x_interval_MD = [(date - examdates_all[0]) for date in df_MD_int.Examdate]
                ols2_int = reg_ols(x_interval_MD, df_MD_int['MD'], tail='right')
                y_pred_interval_md = [ols2_int.intercept + ols2_int.slope * x for x in interval]
            else:
                ols2_int = pd.Series({'slope': pd.NA, 'pval': pd.NA, 'intercept': pd.NA, 'r2': pd.NA})

            ax1.plot(interval, y_pred_interval, label=f'Interval {i + 1}: {ols0_int.slope:.3f} /y, pval: {ols0_int.pval:.2f}, r2: {ols0_int.r2:.2f}', linestyle=':', color=line_colors[i])
            output.append(generate_output_series(ols0_int, ols2_int, df_clean_int, df_excl_int, examdates_clean, case, eye))

    ax1.plot(x, coef_ols0, c='b', label=f'G-RISK (reg0): {round(ols0["slope"], 3)} /y, pval: {round(ols0["pval"], 2)}, r2: {round(ols0["r2"], 2)}', alpha=0.8)
    ax1.scatter(examdates_clean, df_clean['G-RISK_reg1'], marker='^', c='none', edgecolors='black')
    ax1.scatter(examdates_excl, df_excl['G-RISK_reg1'], marker='^', c='none', edgecolors='red')
    ax1.set_ylim([0, 1.1])
    ax1.set_facecolor('#F8F5FF')
    ax1.grid(axis='y', alpha=.4, linewidth=.8)
    ax1.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=True)
    ax1.legend(loc='lower left', fontsize=6)

    fig.tight_layout()
    fig.savefig(f'./progplots/{case}/grisk_trend/{case}_{eye}_grisk_qle{QUALITY_EXCL}_es{EYESIDE_EXCL}_sh{SHAPE_EXCL}_dd{DISCD_DETECT_EXCL}_hu{DISCHU_EXCL}_ymin{M}_ymax{N}_vis{O}_mdiff{MAX_DIFF}.pdf', pad_inches=0)
    plt.close(fig)

    if not INTERVAL:
        output.append(generate_output_series(ols0, pd.Series(), df_clean, df_excl, examdates_clean, case, eye))
    return output

def main():
    od_info_files = [os.path.join(path, name).replace("\\", "/")
                     for path, _, files in os.walk('./progplots/')
                     for name in files if 'ODinfo' in name]

    data = []
    for f in tqdm(od_info_files):
        case = get_case(f)
        eye = get_eye(f)
        output_path = f'./progplots/{case}/grisk_trend/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        try:
            data.append(grisk_progplot(case, eye))
        except Exception as e:
            print(f"Error processing {f}: {e}")
            continue

    columns = ['case', 'eye', 'age', 'baseline_grisk_reg0', 'grisk_slope_reg0', 'grisk_pval_reg0', 'r2_reg0', 'visits', 'FU_years', 'img', 'img_examdate', 'removal%', 'mean_hu', 'mean_dr', 'std_quality', 'std_eyeside', 'timebetwvis']
    df = pd.DataFrame(columns=columns)

    for sublist in data:
        flattened_list = pd.DataFrame(sublist)
        df = pd.concat([df, flattened_list], ignore_index=True)

    df.to_excel(f'./proglabels/labels_qle{QUALITY_EXCL}_es{EYESIDE_EXCL}_sh{SHAPE_EXCL}_dd{DISCD_DETECT_EXCL}_hu{DISCHU_EXCL}_ymin{M}_ymax{N}_vis{O}_mdiff{MAX_DIFF}.xlsx', index=False)

if __name__ == '__main__':
    main()
