API reference
=============

The supported user-facing surface is intentionally small. Modules not listed
here are implementation details and may change during the 0.x series.

Top-level API
-------------

.. autofunction:: neuroflow.load
.. autofunction:: neuroflow.open_source
.. autofunction:: neuroflow.open_dandi
.. autofunction:: neuroflow.open_array
.. autofunction:: neuroflow.open_result
.. autofunction:: neuroflow.plan
.. autofunction:: neuroflow.run
.. autofunction:: neuroflow.compare_segmentations

Named arrays
------------

.. autoclass:: neuroflow.NeuroArray
   :members:

Adapters
--------

.. automodule:: neuroflow.adapters
   :members:

Partition plans
---------------

.. automodule:: neuroflow.partition
   :members:

Selection
---------

.. automodule:: neuroflow.selection
   :members:

Cellpose integration
--------------------

.. automodule:: neuroflow_cellpose
   :members:

Pynapple integration
--------------------

.. automodule:: neuroflow_pynapple
   :members:
