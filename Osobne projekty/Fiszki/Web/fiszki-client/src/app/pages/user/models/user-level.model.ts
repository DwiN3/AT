export class UserLevelModel
{
    level : number;
    nextLevelPoints : number;
    points : number;

    constructor(level : number, nextLevelPoints : number, points : number)
    {
        this.level = level;
        this.nextLevelPoints = nextLevelPoints;
        this.points = points;
    }
}