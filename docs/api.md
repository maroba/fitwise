# API reference

## Fitting

```{eval-rst}
.. autofunction:: fitwise.fit
```

## Results

```{eval-rst}
.. autoclass:: fitwise.FitResult
   :members:

.. autoclass:: fitwise.Parameter
   :members:

.. autoclass:: fitwise.Parameters
   :members:
```

## Models

```{eval-rst}
.. autoclass:: fitwise.Model
   :members:

.. autoclass:: fitwise.ExpressionModel
   :members:

.. autodata:: fitwise.expression.FUNCTIONS
   :no-value:

.. autoclass:: fitwise.CompositeModel
   :members:
```

### Built-in models

```{eval-rst}
.. automodule:: fitwise.models.basic
   :members:

.. automodule:: fitwise.models.peaks
   :members:

.. automodule:: fitwise.models.curves
   :members:
```

## Warnings

```{eval-rst}
.. automodule:: fitwise.warnings
   :members:
```

## Output

```{eval-rst}
.. autofunction:: fitwise.report.format_measurement

.. autofunction:: fitwise.plotting.plot_result

.. autofunction:: fitwise.serialize.to_dict
```
