import { Injectable } from "@angular/core";
import { ActivatedRouteSnapshot, CanActivate, Router, RouterStateSnapshot, UrlTree } from "@angular/router";
import { catchError, map, Observable, of } from "rxjs";
import { AccountService } from "./user.service";

@Injectable()
export class AuthGuard implements CanActivate {
    
    constructor(private accountService : AccountService, private router : Router){}

    canActivate(route: ActivatedRouteSnapshot, state: RouterStateSnapshot): boolean | UrlTree | Observable<boolean | UrlTree> | Promise<boolean | UrlTree> {
        return this.accountService.CheckTokenValidity().pipe(
            map(status => {
                return true;
            }),
            catchError(error => {
                this.router.navigate(['']);
                return of(false);
            })
        )
                
    }

}