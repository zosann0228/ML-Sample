
#include"ml.h"

double calculateUnit(Unit const* unit, Vector const* x)
{
  // create a copy of x to contain calculate result
  Vector* _x = copyVector(x);

  // multiply the x by weight
  vectorMul(_x, unit->a);
  // Activate
  double y = unit->f(vectorSum(_x) + unit->b);

  // destroy the copy of x
  destroyVector(_x);
  return y;
}

double calculateUnitError(Unit const* unit, Vector const* x, double y)
{
  double e = (y - calculateUnit(unit, x));
  return e * e;
}

double unit_delta = 0.001;

Unit* trainUnit(Unit* unit, Vector const* x, const double y, const double rate)
{
  Unit* _delta_unit = copyUnit(unit);
  Unit* _new_unit = copyUnit(unit);

  for(size_t n = 0; n < x->dimention; n++)
  {
    _delta_unit->a->value[n] += unit_delta;
    _new_unit->a->value[n] -= ((calculateUnitError(_delta_unit, x, y) - calculateUnitError(unit, x, y)) / unit_delta) * rate;
    // Revert
    _delta_unit->a->value[n] -= unit_delta;
  }
  _delta_unit->b += unit_delta;
  _new_unit->b -= ((calculateUnitError(_delta_unit, x, y) - calculateUnitError(unit, x, y)) / unit_delta) * rate;

  // destroy the delta unit
  destroyUnit(_delta_unit);

  double err[2] = {calculateUnitError(unit, x, y), calculateUnitError(_new_unit, x, y)};

  if(err[0] > err[1])
  {
    destroyUnit(unit);
    return _new_unit;
  }

  destroyUnit(_new_unit);
  return unit;
}
