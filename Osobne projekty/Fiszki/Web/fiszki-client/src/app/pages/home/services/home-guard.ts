import { Injectable } from "@angular/core";
import { ActivatedRouteSnapshot, CanActivate, Router, RouterStateSnapshot, UrlTree } from "@angular/router";
import { catchError, map, Observable, of } from "rxjs";
import { AccountService } from "src/app/shared/services/user.service";

@Injectable()
export class HomeGuard implements CanActivate {
    
    constructor(private accountService : AccountService, private router : Router){}

    canActivate(route: ActivatedRouteSnapshot, state: RouterStateSnapshot): boolean | UrlTree | Observable<boolean | UrlTree> | Promise<boolean | UrlTree> {
        return this.accountService.CheckTokenValidity().pipe(
            map(status => {
                this.router.navigate(['/user'])
                return false;
            }),
            catchError(error => {
                return of(true);
            })
        )
                
    }

}