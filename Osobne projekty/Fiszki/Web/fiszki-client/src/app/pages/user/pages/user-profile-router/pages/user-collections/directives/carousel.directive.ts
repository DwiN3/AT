import { Directive, ElementRef, HostListener, OnDestroy, OnInit, Renderer2 } from "@angular/core";
import { Store } from "@ngrx/store";
import { Subject, takeUntil } from "rxjs";
import { CarouselSettingsService } from "../services/carousel-settings.service";
import { resetPage, setElementsToDisplay } from "../store/carousel.actions";
import { CarouselState } from "../store/carousel.state";

@Directive({
    selector: '[carousel]',
})
export class CarouselDirective implements OnInit, OnDestroy
{

    margin : number = 0;
    carouselWidth : number = 0;
    elementsQuantity : number = 0;
    elementsToDisplay : number = 0;
    currentPage : number = 0;

    private destroy$ = new Subject<void>();

    constructor(private el : ElementRef, private renderer : Renderer2, private store : Store<{carousel : CarouselState}>,  private carouselSettingsService :   CarouselSettingsService)
    {
        this.store.select('carousel')
            .pipe(takeUntil(this.destroy$))
            .subscribe(data => {
                this.currentPage = data.currentPage;
                this.elementsQuantity = data.collectionQuantity;
                this.elementsToDisplay = data.elementsToDisplay;
                this.HandlePageChange();
                this.SetWidth();
                this.SetStyle();
            });

    }
    
    ngOnInit(): void {

        const carouselSettings = this.carouselSettingsService.SetQuantity(this.elementsQuantity);
        
        this.store.dispatch(setElementsToDisplay({value : carouselSettings.elementsToDisplay, pageQuantity : carouselSettings.pageQuantity})); 

    }

    ngOnDestroy(): void {
        this.destroy$.next();
        this.destroy$.complete();
    }

    private SetStyle() : void
    {
        this.renderer.setStyle(this.el.nativeElement, 'width', this.carouselWidth + 'px');
    }

    private SetWidth() : void
    {
        const boxWidth = (window.innerWidth * 90 / 100) * 90 / 100;
        const flashcardWidth = 200;

        this.margin = (boxWidth - this.elementsToDisplay * flashcardWidth) / (this.elementsToDisplay - 1);
        this.carouselWidth = this.elementsQuantity * 200 + (this.elementsQuantity - 1)  * this.margin;
    }

    private HandlePageChange(): void 
    {
        const slideValue =  (this.currentPage * 200 + this.margin * this.currentPage) * -1;

        this.renderer.setStyle(this.el.nativeElement, 'transform', 'translateX(' + slideValue + 'px)');
    }

    @HostListener('window:resize', ['$event'])
    onResize(event : any) : void
    {
        const carouselSettings = this.carouselSettingsService.SetQuantity(this.elementsQuantity);

        this.store.dispatch(setElementsToDisplay({value : carouselSettings.elementsToDisplay, pageQuantity : carouselSettings.pageQuantity}));
        this.store.dispatch(resetPage());

        this.SetWidth();
        this.SetStyle();      
    }
}