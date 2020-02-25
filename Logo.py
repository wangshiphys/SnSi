"""
Generate the logo of this project.
"""


import matplotlib.pyplot as plt

fig = plt.figure()
fig.text(
    0.5, 0.4, "SnSi", va="center", ha="center",
    fontdict={"size": 80, "style": "italic"},
)
fig.set_size_inches(2.65, 0.92)
fig.savefig("icons/logo.svg")
plt.close("all")
