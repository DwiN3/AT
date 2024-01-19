export class BaseCurrentUserModel
{
    firstName : string;
    lastName : string;

    constructor(name : string, lastname : string)
    {
        this.firstName = name;
        this.lastName = lastname;
    }
}