class Animal {
    name: string;
    constructor(name: string){
        this.name = name;
    }

    makeSound(): string {
        return "";
    }
}

class Dog extends Animal {
    constructor(name: string) {
        super(name);
    }

    makeSound(): string {
        return "Woof";
    }
}

class Cat extends Animal {
    constructor(name: string) {
        super(name);
    }

    makeSound(): string {
        return "Meow";
    }
}

const dog = new Dog("Alex");
console.log(dog.makeSound());

const cat = new Cat("John");
console.log(cat.makeSound());