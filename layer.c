
#include"ml.h"

Vector* calculateLayer(Layer const* layer, Vector const* x)
{
  Vector* y = createVector(layer->unit_count);
  for(size_t n = 0; n < layer->unit_count; n++)
    y->value[n] = calculateUnit(layer->units[n], x);
  return y;
}

Vector* calculateLayerError(Layer const* layer, Vector const* x, Vector const* y)
{
  Vector* err = createVector(layer->unit_count);
  for(size_t n = 0; n < layer->unit_count; n++)
    err->value[n] = calculateUnitError(layer->units[n], x, y->value[n]);
  return err;
}

static inline Vector* maxIndex(Vector const* x)
{
  Vector* y = createVector(x->dimention);

  double last_max = x->value[0];
  for(size_t i = 0; i < x->dimention; i++)
  {
    double* index = &y->value[i];

    int _max = 0;
    double max;
    for(size_t n = 0; n < x->dimention; n++)
    {

      if(last_max > x->value[0])
      {
        if(!_max)
        {
          max = x->value[n];
          *index = (double)n;
          _max = 1;
        }
        else if(max < x->value[n])
        {
          max = x->value[n];
          *index = (double)n;
        }
      }
    }

    last_max = x->value[(size_t)*index];
  }

  return y;
}

Layer* trainLayer(Layer* layer, Vector const* x, Vector const* y, double rate)
{
  Layer* _layer = copyLayer(layer);
  Vector* error = calculateLayerError(layer, x, y);
  double _rate = rate;

  // Get index array that is bigger order
  Vector* errIndex = maxIndex(error);

  // train each units
  for(size_t n = 0; n < errIndex->dimention; n++)
  {
    _layer->units[n] = trainUnit(_layer->units[n], x, y->value[(size_t)errIndex->value[n]], _rate);
    _rate *= rate;
  }

  destroyVector(errIndex);

  Vector* trained_error = calculateLayerError(_layer, x, y);
  Layer* ret_layer;
  if(vectorSum(error) > vectorSum(trained_error))
  {
    ret_layer = _layer;
    destroyLayer(layer);
  }
  else
  {
    ret_layer = layer;
    destroyLayer(_layer);
  }

  destroyVector(error);
  destroyVector(trained_error);

  return ret_layer;
}
