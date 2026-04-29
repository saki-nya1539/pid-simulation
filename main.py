import numpy as np
import matplotlib.pyplot as plt


def simulate_pid(Kp, Ki, Kd, target=1.0, T=1.0, dt=0.01, total_time=10.0):
    time = np.arange(0, total_time, dt)

    y = 0.0
    integral = 0.0
    previous_error = 0.0

    outputs = []

    for _ in time:
        error = target - y

        integral += error * dt
        derivative = (error - previous_error) / dt

        u = Kp * error + Ki * integral + Kd * derivative

        y = y + ((-y + u) / T) * dt

        outputs.append(y)
        previous_error = error

    return time, np.array(outputs)


def calculate_metrics(time, output, target):
    max_value = np.max(output)

    overshoot = max(0, (max_value - target) / target * 100)

    tolerance = 0.02 * target
    settling_time = None

    for i in range(len(output)):
        if np.all(np.abs(output[i:] - target) <= tolerance):
            settling_time = time[i]
            break

    steady_state_error = abs(target - output[-1])

    return overshoot, settling_time, steady_state_error


target = 1.0

params_list = [
    {"Kp": 1.0, "Ki": 0.0, "Kd": 0.0, "label": "P control"},
    {"Kp": 2.0, "Ki": 0.5, "Kd": 0.0, "label": "PI control"},
    {"Kp": 2.0, "Ki": 0.5, "Kd": 0.2, "label": "PID control"},
]

plt.figure(figsize=(10, 5))

for params in params_list:
    time, output = simulate_pid(
        params["Kp"],
        params["Ki"],
        params["Kd"],
        target=target
    )

    overshoot, settling_time, steady_state_error = calculate_metrics(
        time,
        output,
        target
    )

    print("===" + params["label"] + "===")
    print(f"Kp={params['Kp']}, Ki={params['Ki']}, Kd={params['Kd']}")
    print(f"オーバーシュート: {overshoot:.2f}%")

    if settling_time is None:
        print("収束時間: 未収束")
    else:
        print(f"収束時間: {settling_time:.2f} 秒")

    print(f"定常偏差: {steady_state_error:.4f}")
    print()

    plt.plot(
        time,
        output,
        label=f"{params['label']} Kp={params['Kp']}, Ki={params['Ki']}, Kd={params['Kd']}"
    )

plt.axhline(target, color="red", linestyle="--", label="Target")
plt.xlabel("Time [s]")
plt.ylabel("Output")
plt.title("P / PI / PID Comparison")
plt.legend()
plt.grid(True)
plt.show()