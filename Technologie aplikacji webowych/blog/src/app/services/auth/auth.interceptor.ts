import { HttpInterceptorFn } from '@angular/common/http';

export const authInterceptor: HttpInterceptorFn = (req, next) => {

  const token = localStorage.getItem('token');

  if (token) {
    const authReq = req.clone({
      setHeaders: {
        'x-auth-token': `Bearer ${token}`
      }
    });
    return next(authReq);
  } else {
    return next(req);
  }

}