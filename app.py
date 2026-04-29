import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def simulate_pid(Kp, Ki, Kd, target=1.0, T=1.0, dt=0.01, total_time=10.0):
    time = np.arange(0, total_time, dt)

    y = 0.0
    integral = 0.0
    previous_error = 0.0

    outputs = []
    controls = []

    for _ in time:
        error = target - y

        integral += error * dt
        derivative = (error - previous_error) / dt

        u = Kp * error + Ki * integral + Kd * derivative

        y = y + ((-y + u) / T) * dt

        outputs.append(y)
        controls.append(u)
        previous_error = error

    return time, np.array(outputs), np.array(controls)


def calculate_metrics(time, output, target):
    max_value = np.max(output)

    if target == 0:
        overshoot = 0
        tolerance = 0.02
    else:
        overshoot = max(0, (max_value - target) / target * 100)
        tolerance = 0.02 * abs(target)

    settling_time = None

    for i in range(len(output)):
        if np.all(np.abs(output[i:] - target) <= tolerance):
            settling_time = time[i]
            break

    steady_state_error = abs(target - output[-1])

    return overshoot, settling_time, steady_state_error


st.title("PID制御シミュレーション可視化ツール")

st.write("P/I/Dゲインを変更し、制御対象の応答の違いを確認できます。")

st.sidebar.header("パラメータ設定")

target = st.sidebar.slider("目標値", 0.0, 5.0, 1.0, 0.1)
Kp = st.sidebar.slider("Kp 比例ゲイン", 0.0, 10.0, 2.0, 0.1)
Ki = st.sidebar.slider("Ki 積分ゲイン", 0.0, 5.0, 0.5, 0.1)
Kd = st.sidebar.slider("Kd 微分ゲイン", 0.0, 5.0, 0.1, 0.1)
T = st.sidebar.slider("時定数 T", 0.1, 5.0, 1.0, 0.1)
total_time = st.sidebar.slider("シミュレーション時間", 1.0, 30.0, 10.0, 1.0)

time, output, control = simulate_pid(
    Kp=Kp,
    Ki=Ki,
    Kd=Kd,
    target=target,
    T=T,
    total_time=total_time
)

overshoot, settling_time, steady_state_error = calculate_metrics(time, output, target)

st.subheader("評価指標")

col1, col2, col3 = st.columns(3)

col1.metric("オーバーシュート", f"{overshoot:.2f}%")

if settling_time is None:
    col2.metric("収束時間", "未収束")
else:
    col2.metric("収束時間", f"{settling_time:.2f} s")

col3.metric("定常偏差", f"{steady_state_error:.4f}")

st.subheader("応答グラフ")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(time, output, label="Output")
ax.axhline(target, color="red", linestyle="--", label="Target")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Output")
ax.set_title("PID Response")
ax.grid(True)
ax.legend()

st.pyplot(fig)

st.subheader("制御入力グラフ")

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(time, control, color="orange", label="Control Input")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Input")
ax2.set_title("Control Input")
ax2.grid(True)
ax2.legend()

st.pyplot(fig2)

st.subheader("考察メモ")
st.write("""
- Kpを大きくすると応答速度は上がりやすいが、オーバーシュートが増える場合がある。
- Kiを加えると定常偏差を小さくできるが、振動しやすくなる場合がある。
- Kdを加えると急激な変化を抑え、応答を安定させる効果がある。
""")