"""Force a non-interactive matplotlib backend for notebook tests.

Several tests exec notebook cells in-process; with an interactive backend
(e.g. TkAgg on a desktop session) ``plt.show()`` would block forever in the
GUI mainloop.
"""

import matplotlib

matplotlib.use("Agg", force=True)
