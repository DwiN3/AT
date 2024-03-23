export class Point {
    x:number;
    y:number;

    constructor(x:number, y:number){
        this.x = x;
        this.y = y;
    }

    move(x:number, y:number) : void {
        this.x += x;
        this.y += y;
    }
}

export class Rectangle {
    constructor(
        public topLeft: Point,
        public topRight: Point,
        public bottomLeft: Point,
        public bottomRight: Point
    ) {}

    move(x: number, y: number): void {
        this.topLeft.move(x, y);
        this.topRight.move(x, y);
        this.bottomLeft.move(x, y);
        this.bottomRight.move(x, y);
    }
    
    getArea(): number {
        const width = Math.abs(this.topRight.x - this.topLeft.x);
        const height = Math.abs(this.topLeft.y - this.bottomLeft.y);
        return width * height;
    }
}

const topLeft = new Point(1, 1);
const topRight = new Point(5, 1);
const bottomLeft = new Point(1, 4);
const bottomRight = new Point(5, 4);
const rectangle = new Rectangle(topLeft, topRight, bottomLeft, bottomRight);

console.log('\nPole = '+rectangle.getArea());

console.log('Punkty:                 ',rectangle.topLeft);
rectangle.move(2, 2);
console.log('Punkty po przesunięciu: ',rectangle.topLeft);