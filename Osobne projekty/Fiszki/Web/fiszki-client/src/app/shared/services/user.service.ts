import { Injectable } from "@angular/core";
import { BaseUserModel } from "../models/base-user.model";
import { HttpClient, HttpHeaders } from "@angular/common/http";
import { RegisterUserModel } from "../models/register-user.model";

@Injectable({providedIn : 'root'})
export class AccountService
{
    url="http://localhost:8080/flashcards/auth/";
    constructor(private http : HttpClient){}

    Login(userData : BaseUserModel)
    {
        return this.http.post<any>(this.url + 'login', userData);
    }

    Register(userData : RegisterUserModel)
    {
        return this.http.post<any>(this.url + 'register', userData);
    }

    CheckTokenValidity()
    {
        const token = localStorage.getItem('token');

        const headers = new HttpHeaders({
            'Authorization' : `Bearer ${token}`
        })

        return this.http.post<any>(this.url + 'access', { token : token}, {headers : headers});
    }

    LogOut() : void
    {
        localStorage.removeItem('token');
    }
}