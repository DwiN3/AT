export class AlertModel
{
    title : string;
    details : string;
    instructions : string;

    constructor(title : string, details : string, instructions : string)
    {
        this.title = title;
        this.details = details;
        this.instructions = instructions;
    }
}