Installation & Requirements
===========================

Requirements
------------
PyFC requires **Python 3.8+**. Core dependencies include:

* ``numpy >= 1.20``
* ``scipy``
* ``numba``
* ``matplotlib``

**Optional Dependencies:**

* ``ultranest`` (Required for utilizing the nested sampling optimizer)
* ``tqdm`` (Provides progress bars during execution)

Installation
------------
**Option 1: Direct PyPI Installation (Recommended)**

To install the latest stable release directly from the Python Package Index (PyPI), run:

.. code-block:: bash

   pip install PyFeldmanCousins

``pip`` is not case-sensitive, so using ``pip install pyfeldmancousins`` will have the same result.

**Option 2: Local Installation from Source**

If you want to run the latest development version or modify the codebase, clone the repository and install via ``pip`` to automatically resolve and install all dependencies:

.. code-block:: bash

   git clone https://github.com/mbustama/FeldmanCousins.git
   cd FeldmanCousins
   pip install -e .

Verifying the Installation
--------------------------
After installing the package, you can run the local test suite to ensure everything is configured correctly for your system architecture. We utilize ``pytest`` to validate statistical correctness, optimizer behavior, and parallelization scaling.

.. code-block:: bash

   # Install the testing framework
   pip install pytest

   # Run the test suite from the repository root
   pytest tests/ -v

Developer Installation
----------------------
If you plan to modify the codebase or contribute to the project, you should install the package with its optional testing and development dependencies included:

.. code-block:: bash

   pip install -e ".[test]"