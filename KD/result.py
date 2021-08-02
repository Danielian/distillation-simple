from pyecharts.charts import Line
import pyecharts.options as opts
import matplotlib.pyplot as plt

def result_draw(t_data, kd_data, s_data, epochs):
    x = list(range(1, epochs + 1))
    plt.subplot(2, 1, 1)
    plt.plot(x, [t_data[i][1] for i in range(epochs)], label='teacher')
    plt.plot(x, [kd_data[i][1] for i in range(epochs)], label='student with KD')
    plt.plot(x, [s_data[i][1] for i in range(epochs)], label='student without KD')

    plt.title('Test accuracy')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x, [t_data[i][0] for i in range(epochs)], label='teacher')
    plt.plot(x, [kd_data[i][0] for i in range(epochs)], label='student with KD')
    plt.plot(x, [s_data[i][0] for i in range(epochs)], label='student without KD')

    plt.title('Test loss')
    plt.legend()
    plt.savefig(fname="result/result.png")
    plt.show()
