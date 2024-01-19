import { Injectable } from "@angular/core";
import { ActivatedRouteSnapshot, Resolve, RouterStateSnapshot } from "@angular/router";
import { Observable } from "rxjs";
import { UserLevelModel } from "../models/user-level.model";
import { UserService } from "./user.service";

@Injectable()
export class UserLevelResolver implements Resolve<UserLevelModel>
{
    constructor(private userService : UserService){}

    resolve(route: ActivatedRouteSnapshot, state: RouterStateSnapshot): UserLevelModel | Observable<UserLevelModel> | Promise<UserLevelModel> {
        return this.userService.GetUserLevel();
    }

}