=====================================================================
Welcome to CS2D's documentation!
=====================================================================

- **Data-driven methods to select samples**
- **A benchmark for backdoor attack**


Check out the :doc:`gettingstarted` section for further information, including
how to install the project.

.. note::

   This project is under active development.

------------------
An example
------------------

Here's a demonstrating example of how to use the package:




.. code-block:: python


   import cs2d
   import cs2d.models as c2m
   import cs2d.datasets as csd
   
   ### create data for binary classification
   traindata, trainlabel, testdata, testlabel = csd.linear_data(4000,1000)
   backdoor_target = 1
   number_of_bd = 50
   
   ### classifier and selection mechanism
   cs2d_classifier = c2m.Linearclassifier()
   cs2d_selection = c2m.Synselection()
   
   ### training 
   cs2d(cs2d_classifier, cs2d_selection).fit(traindata, trainlabel, backdoor_target, number_of_bd)
   
   ### test for selection
   cs2d.backdoor_selection(testdata, cutoff = 0.1)
   
   ### test for accuracy
   cs2d.backdoor_score(test, backdoor_target, cutoff = 0.1)
   
   
   
    
  
   
   

   





Table of Contents
=================
.. toctree::
   :maxdepth: 2
    
   Getting Started <gettingstarted>
   APIs <api>
   Gallery <gallery>




Advantages of using CS2D
==========================
- **End-to-end learning** - The CS2D provides end-to-end learning.


Dependencies
============

CS2D has minimal dependencies, with core functionality being pure python:

- shapely
- pandas
- numpy
- requests
- pint

Contact
=======

Xun Xian, ECE, University of Minneosta, xian0044@umn.edu

Jie Ding, School of Statistics, University of Minnesota, dingj@umn.edu



Important Links
===============



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
