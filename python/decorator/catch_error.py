def catch_error(func):
  def _f(*args, **kargs):
    try:
      return func(*args, **kargs)
    except Exception:
      print('catch error')
  return _f

@catch_error
def r(n):
  if n == 10:
    raise Exception

  return n

print(r(20))
print(r(10))
