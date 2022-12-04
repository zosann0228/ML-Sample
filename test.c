
#include<stdio.h>
#include<stdlib.h>

#include<time.h>

#include"ml.h"

double f(double x)
{
  return x;
}

double ReLU(double x)
{
  return max(x, 0);
}

double tf(double x)
{
  return x;
}

double tf2(double x)
{
  return x * x;
}

double maxDoubleArray(double* x, size_t n)
{
  if(n > 0)
  {
    double m = x[0];
    for(size_t p = 0; p < n; p++)
      if(x[p] > m)
        m = x[p];
    return m;
  }
  return 0;
}

double minDoubleArray(double* x, size_t n)
{
  if(n > 0)
  {
    double m = x[0];
    for(size_t p = 0; p < n; p++)
      if(x[p] < m)
        m = x[p];
    return m;
  }
  return 0;
}

struct standarizedata
{
  double max;
  double min;
};

struct standarizedata standarize(double* x, size_t n)
{
  struct standarizedata d;
  d.max = maxDoubleArray(x, n);
  d.min = minDoubleArray(x, n);

  for(int p = 0; p < n; p++)
    x[p] = (x[p] - d.min) / (d.max - d.min);
  return d;
}

double unstandarize(double x, struct standarizedata d)
{
  return x * (d.max - d.min) + d.min;
}

double* shuffleto(double* x, size_t n)
{
  double* y = amalloc(sizeof(double) * n);
  for(size_t p = 0; p < n; p++)
    y[p] = x[rand() % n];
  return y;
}

double begin = 0;
double rate = 0.01;
size_t epoch = 10;

struct standarizedata prepare(double* x, double* y, size_t n, double (*test_func)(double x))
{
  for(int p = 0; p < n; p++)
  {
    x[p] = begin + p * rate;
    y[p] = test_func(x[p]);
  }

  struct standarizedata d = standarize(x, n);
  
  unsigned int seed = (unsigned int)time(NULL);
  srand(seed);
  double* ix = shuffleto(x, n);
  memcpy(x, ix, sizeof(double) * n);
  srand(seed);
  double* iy = shuffleto(y, n);
  memcpy(y, iy, sizeof(double) * n);

  free(ix);
  free(iy);
}

int main(int argc, char* argv[])
{
  size_t n = 1000;
  double* x = amalloc(sizeof(double) * n);
  double* y = amalloc(sizeof(double) * n);

  struct standarizedata d = prepare(x, y, n, tf);

  Unit* u = createUnit(1, f);
  Vector* vx = createVector(1);
  for(size_t t = 0; t < 1; t++)
    for(size_t p = 0; p < n; p++)
    {
      vx->value[0] = x[p];
      u = trainUnit(u, vx, y[p], 0.05);
      if((p % epoch) == 0)
        printf("[%d] error:%f\r\n", p, calculateUnitError(u, vx, y[p]));
    }

  for(size_t p = 0; p < 10; p++)
  {
    vx->value[0] = y[p];
    printf("f(%f) = %f\r\n", unstandarize(vx->value[0], d), calculateUnit(u, vx));
  }

  destroyUnit(u);

  // Test 2
  printf("\r\n-----\r\nTest2\r\n-----\r\n");

  d = prepare(x, y, n, tf2);

  Vector* a_vy = createVector(1);

  Layer* l[2];
  l[0] = createLayer(10, 1, ReLU);
  l[1] = createLayer(1, 10, f);

  return 0;
}
