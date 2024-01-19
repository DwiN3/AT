import { HttpClient, HttpHeaders } from "@angular/common/http";
import { Injectable } from "@angular/core";
import { UserPasswordInterface } from "src/app/shared/models/user-password.interface";
import { UserLevelModel } from "../models/user-level.model";
import { UserPasswordExtendInterface } from "../pages/user-profile-router/pages/user-change-password/models/user-password-extend";

@Injectable({providedIn : 'root'})
export class UserService
{
    url : string = "http://localhost:8080/flashcards/auth/";

    constructor(private http : HttpClient){}

    GetUserInfo()
    {
        const token = localStorage.getItem('token');
        const headers = new HttpHeaders({
            'Authorization' : `Bearer ${token}`
        })

        return this.http.get<any>(this.url + 'info', {headers : headers});
    }

    GetUserLevel()
    {
        const token = localStorage.getItem('token');
        const headers = new HttpHeaders({
            'Authorization' : `Bearer ${token}`
        })

        return this.http.get<UserLevelModel>(this.url + 'level', {headers : headers});
    }

    AddResult(points : number)
    {
        const token = localStorage.getItem('token');
        const headers = new HttpHeaders({
            'Authorization' : `Bearer ${token}`
        })
        return this.http.put<any>(this.url + 'points', {points : points}, {headers : headers});
    }

    ChangePassword(data : UserPasswordExtendInterface)
    {
        return this.http.put<any>(this.url + 'change-password', data);
    }

    ResetPassword(data : UserPasswordInterface)
    {
        return this.http.put<any>(this.url + 'process-password-change', data)
    }

}