import { Inject, Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { JwtHelperService } from '@auth0/angular-jwt';
import { map } from 'rxjs/operators';
import { Token } from '../../models/token';
import { DOCUMENT } from '@angular/common';
import { jwtDecode } from 'jwt-decode';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private url = 'http://localhost:3001/api';

  constructor(
    private http: HttpClient,
    @Inject(DOCUMENT) private document: Document
  ) {}

  authenticate(credentials: any) {
    const localStorage = this.document.defaultView?.localStorage;
    return this.http.post<Token>(`${this.url}/user/auth`, {
      login: credentials.login,
      password: credentials.password
    }).pipe(
      map((result: Token) => {
        if (result && result.token) {
          localStorage?.setItem('token', result.token);
          const decodedToken: any = jwtDecode(result.token);
          const role = decodedToken?.role || 'user';
          const email = decodedToken?.name || '';
          const username = decodedToken?.username || '';
          localStorage?.setItem('role', role);
          localStorage?.setItem('email', email);
          localStorage?.setItem('username', username);
          return true;
        }
        return false;
      })
    );
  }

  createOrUpdate(credentials: any) {
    return this.http.post(this.url + '/user/create', credentials);
  }

  logout() {
    const localStorage = this.document.defaultView?.localStorage;
    return this.http.delete(this.url + '/user/logout/' + this.currentUser.userId)
      .pipe(
        map(() => {
          localStorage?.removeItem('username');
          localStorage?.removeItem('token');
          localStorage?.removeItem('role');
        })
      );
  }

  deleteAccount() {
    const localStorage = this.document.defaultView?.localStorage;
    const userId = this.currentUser?.userId;
    return this.http.delete(`${this.url}/user/${userId}`).pipe(
      map(() => {
        localStorage?.clear();
      })
    );
  }

  changePassword(newPassword: string) {
    const userId = this.currentUser?.userId;
    return this.http.post(`${this.url}/user/change-password/${userId}`, { newPassword });
  }

  isLoggedIn() {
    const localStorage = this.document.defaultView?.localStorage;
    const jwtHelper = new JwtHelperService();
    const token = localStorage?.getItem('token');
    if (!token) {
      return false;
    }
    return !jwtHelper.isTokenExpired(token);
  }

  get currentUser() {
    const token = this.getToken();
    if (!token) {
      return null;
    }
    return new JwtHelperService().decodeToken(token);
  }

  getToken() {
    const localStorage = this.document.defaultView?.localStorage;
    return localStorage?.getItem('token');
  }

  getUserName(): string | null {
    const localStorage = this.document.defaultView?.localStorage;
    return localStorage?.getItem('username') ?? null;
  }

  isAdmin(): boolean {
    const localStorage = this.document.defaultView?.localStorage;
    return localStorage?.getItem('role') === 'admin';
  }

  getEmail(): string | null {
    const localStorage = this.document.defaultView?.localStorage;
    return localStorage?.getItem('email') ?? null;
  }
}
