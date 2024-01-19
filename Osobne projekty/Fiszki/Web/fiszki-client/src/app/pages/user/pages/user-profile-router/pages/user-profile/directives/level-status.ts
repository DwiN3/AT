import { Directive, ElementRef, Input, OnInit, Renderer2 } from "@angular/core";

@Directive({
    selector: '[lvlStatus]'
})
export class LevelDirective implements OnInit
{
    @Input() lvlStatus : number = 0;
    @Input() pointsRequired : number = 0;

    constructor(private el: ElementRef, private renderer: Renderer2){}
    
    ngOnInit(): void {
        this.SetStyle();
    }

    private SetStyle() : void 
    {
        this.renderer.setStyle(this.el.nativeElement, 'width', this.calcLvl() + '%');
    }

    private calcLvl() : number 
    {
        let result = this.lvlStatus / this.pointsRequired * 100;
        return(Math.ceil(result));
    }

}