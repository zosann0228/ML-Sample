#ifndef _ML_H_
#define _ML_H_

#include<stdlib.h>
#include<string.h>

// abort if malloc failed
static inline void* amalloc(size_t size)
{
  void* addr = malloc(size);
  if(addr == NULL)
    abort();
}

typedef struct Vector
{
  double* value;
  size_t dimention;
} Vector;

typedef enum MlResult
{
  mrSuccess,
  mrErrorVector
} MlResult;

static inline Vector* createVector(size_t dimention)
{
  // size of Vector::value
  size_t value_size = sizeof(double) * dimention;
  // allocate memory region to contain a Vector and its value
  void* addr = amalloc(sizeof(Vector) + value_size);
  // Set start addresses
  Vector* vector = (Vector*)addr;
  double* value = (double*)((uintptr_t)addr + sizeof(Vector));

  // 0 clear the value
  memset(value, 0, value_size);

  vector->value = value;
  vector->dimention = dimention;
  return vector;
}

static inline Vector* copyVector(Vector const* vector)
{
  Vector* copy = createVector(vector->dimention);
  // copy all value
  memcpy(copy->value, vector->value, sizeof(double) * vector->dimention);
  return copy;
}

static inline void destroyVector(Vector* v)
{
  free(v);
}

static inline MlResult vectorAdd(Vector* v1, Vector const* v2)
{
  if(v1->dimention != v2->dimention)
    return mrErrorVector;
  for(size_t n = 0; n < v1->dimention; n++)
    v1->value[n] += v2->value[n];
  return mrSuccess;
}

static inline MlResult vectorMul(Vector* v1, Vector const* v2)
{
  if(v1->dimention != v2->dimention)
    return mrErrorVector;
  for(size_t n = 0; n < v1->dimention; n++)
    v1->value[n] *= v2->value[n];
  return mrSuccess;
}

static inline double vectorSum(Vector const* vector)
{
  double y = 0;
  for(int n = 0; n < vector->dimention; n++)
    y += vector->value[n];
  return y;
}

typedef struct Unit
{
  Vector* a;
  double b;
  double (*f)(double x); // Must not NULL
} Unit;

static inline Unit* createUnit(size_t ninputs, double (*f)(double x))
{
  if(f == NULL)
    return NULL;
  
  Vector* a = createVector(ninputs);
  
  Unit* unit = (Unit*)amalloc(sizeof(Unit));

  unit->a = createVector(ninputs);
  unit->b = 0;
  unit->f = f;

  return unit;
}

static inline Unit* copyUnit(Unit const* unit)
{
  Unit* copy = createUnit(unit->a->dimention, unit->f);
  
  memcpy(copy->a->value, unit->a->value, sizeof(double) * unit->a->dimention);
  copy->b = unit->b;

  return copy;
}

static inline void destroyUnit(Unit* unit)
{
  destroyVector(unit->a);
  free(unit);
}

double calculateUnit(Unit const* unit, Vector const* x);
double calculateUnitError(Unit const* unit, Vector const* x, double y);
Unit* trainUnit(Unit const* unit, Vector const* x, double y, double rate);

typedef struct Layer
{
  Unit** units;
  size_t unit_count;
} Layer;

static inline int initLayer(Layer* layer, size_t nunits, size_t ninputs, double (*f)(double x))
{
  layer->units = amalloc(sizeof(Unit*) * nunits);
  for(size_t n = 0; n < nunits; n++)
    layer->units[n] = createUnit(ninputs, f);

  layer->unit_count = nunits;
  return 0;
}

static inline void disposeLayer(Layer* layer)
{
  for(size_t n = 0; n < layer->unit_count; n++)
    destroyUnit(layer->units[n]);
  
  free(layer->units);
}

static inline Layer* createLayer(size_t nunits, size_t ninputs, double (*f)(double x))
{
  Layer* layer = amalloc(sizeof(Layer));

  if(initLayer(layer, nunits, ninputs, f))
  {
    free(layer);
    return NULL;
  }
  return layer;
}

static inline Layer* copyLayer(Layer* layer)
{
  Layer* copy = amalloc(sizeof(Layer));

  copy->units = amalloc(sizeof(Unit*) * layer->unit_count);

  for(size_t n = 0; n < layer->unit_count; n++)
  {
    Unit* unit = layer->units[n];
    copy->units[n] = createUnit(unit->a->dimention, unit->f);
  }

  return copy;
}

static inline void destroyLayer(Layer* layer)
{
  disposeLayer(layer);
  free(layer);
}

Vector* calculateLayer(Layer const* layer, Vector const* x);
Vector* calculateLayerError(Layer const* layer, Vector const* x, Vector const* y);
Layer* trainLayer(Layer* layer, Vector const* x, Vector const* y, double rate);

#endif