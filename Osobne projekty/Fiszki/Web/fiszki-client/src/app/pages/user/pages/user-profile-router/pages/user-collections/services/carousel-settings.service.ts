import { Injectable } from "@angular/core";
import { Store } from "@ngrx/store";
import { CarouselState } from "../store/carousel.state";

@Injectable({providedIn: 'root'})
export class CarouselSettingsService
{
    constructor(private store : Store<{carousel : CarouselState}>){}
    
    SetQuantity(collectionsQuantity : number) : { elementsToDisplay : number, pageQuantity : number}
    {
        const screenWidth : number = window.innerWidth;
        let elementsToDisplay;
        let pageQuantity;

        if(screenWidth < 1050)
            elementsToDisplay = 3;
        else if(screenWidth >= 1050 && screenWidth < 1400)
            elementsToDisplay = 4;
        else if(screenWidth >= 1400 && screenWidth < 1800)
            elementsToDisplay = 5;
        else 
            elementsToDisplay = 6;

        pageQuantity = collectionsQuantity - elementsToDisplay;

        const object = {
            elementsToDisplay : elementsToDisplay,
            pageQuantity : pageQuantity,
        }

        return object;

    }
    

}