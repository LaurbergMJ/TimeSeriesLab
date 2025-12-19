import matplotlib.pyplot as plt 

def plot_lin_reg(test_values, pred_values, label: str | None ):
    plt.figure()
    plt.plot(test_values.index, test_values.values, label=f"{label} actual")
    plt.plot(test_values.index, pred_values, label=f"{label} predicted")
    plt.title(f"Next-day log return prediction")
    plt.legend()
    plt.tight_layout()
    plt.show()



