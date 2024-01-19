import { UserPasswordInterface } from "src/app/shared/models/user-password.interface";

export interface UserPasswordExtendInterface extends UserPasswordInterface 
{
    password : string;
}