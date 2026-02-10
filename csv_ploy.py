import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def plot_latency_analysis(csv_file):
    # 1. 파일 읽기
    if not os.path.exists(csv_file):
        print(f"❌ 오류: '{csv_file}' 파일을 찾을 수 없습니다.")
        return

    try:
        df = pd.read_csv(csv_file)
        df.columns = [c.strip() for c in df.columns]
        
        if 'Time_Sec' not in df.columns or 'Latency_ms' not in df.columns:
            print("⚠️ 컬럼 이름이 맞지 않습니다.")
            return
            
    except Exception as e:
        print(f"❌ 파일 읽기 실패: {e}")
        return

    # 2. 통계 계산 (Pandas Series 사용)
    time_series = df['Time_Sec']
    latency_series = df['Latency_ms']
    
    mean_val = latency_series.mean()
    max_val = latency_series.max()
    std_val = latency_series.std()
    p99_val = latency_series.quantile(0.99)

    print(f"===== 📊 분석 결과 ({os.path.basename(csv_file)}) =====")
    print(f"데이터 개수: {len(df)}개")
    print(f"평균 지연: {mean_val:.3f} ms")
    print(f"최대 지연: {max_val:.3f} ms")
    print(f"표준 편차(Jitter): {std_val:.3f} ms")
    print(f"99% 백분위수: {p99_val:.3f} ms")
    print("===================================")

    # 3. 그래프 그리기 (데이터를 numpy 배열로 변환하여 전달)
    # ★★★ 여기서부터 수정됨 ★★★
    time_data = time_series.to_numpy()
    latency_data = latency_series.to_numpy()

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 4)

    # [왼쪽] 시계열 그래프
    ax1 = fig.add_subplot(gs[0, 0:3])
    ax1.plot(time_data, latency_data, label='Latency', color='#1f77b4', linewidth=0.8, alpha=0.8)
    
    ax1.axhline(mean_val, color='r', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}ms')
    
    threshold = mean_val + 3 * std_val
    spikes = df[df['Latency_ms'] > threshold]
    if not spikes.empty:
        # Scatter plot에도 to_numpy() 적용
        ax1.scatter(spikes['Time_Sec'].to_numpy(), spikes['Latency_ms'].to_numpy(), 
                   color='red', s=15, zorder=5, label='Spikes (>3σ)')

    ax1.set_title(f'System Latency Over Time ({os.path.basename(csv_file)})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Latency (ms)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # [오른쪽] 히스토그램
    ax2 = fig.add_subplot(gs[0, 3], sharey=ax1)
    # 히스토그램에도 to_numpy() 적용
    ax2.hist(latency_data, bins=30, orientation='horizontal', color='#ff7f0e', alpha=0.7, edgecolor='black')
    ax2.axhline(mean_val, color='r', linestyle='--', linewidth=1.5)
    
    ax2.set_title('Distribution', fontsize=12)
    ax2.set_xlabel('Count', fontsize=10)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    save_filename = csv_file.replace('.csv', '_plot.png')
    plt.savefig(save_filename, dpi=300)
    print(f"✅ 그래프 저장 완료: {save_filename}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        target_file = 'latency_result_with_other_process.csv' # 기본값 수정
        
    plot_latency_analysis(target_file)