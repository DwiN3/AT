import { BaseUserModel } from "./base-user.model";

export class RegisterUserModel extends BaseUserModel
{
    firstname : string;
    lastname : string;
    repeatedPassword : string;

    constructor(email : string, password : string, firstname : string, lastname : string, repeatedPassword : string)
    {
        super(email, password);
        this.firstname = firstname;
        this.lastname = lastname;
        this.repeatedPassword = repeatedPassword;
    }
}