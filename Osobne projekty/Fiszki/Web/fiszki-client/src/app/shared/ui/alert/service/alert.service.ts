import { ComponentFactoryResolver, Injectable, ViewChild } from "@angular/core";
import { Subscription } from "rxjs";
import { AlertModel } from "src/app/shared/models/alert.model";
import { AlertComponent } from "../alert.component";
import { PlaceholderDirective } from "../directive/placeholder.directive";

@Injectable({providedIn : 'root'})
export class AlertService
{
    alertSub : Subscription | null = null;
    
    constructor(private componentFactoryResolver : ComponentFactoryResolver){}
    
   ShowAlert(error : string, details : string, instructions : string, alertHost : PlaceholderDirective): void
  {
    const alertCmpFactory = this.componentFactoryResolver.resolveComponentFactory(AlertComponent);
    
    const hostViewContainerRef = alertHost?.viewContainerRef;
    hostViewContainerRef?.clear();

    const componentRef = hostViewContainerRef?.createComponent(alertCmpFactory);

    componentRef.instance.title = error;
    componentRef.instance.instructions = instructions;
    componentRef.instance.details = details;

    this.alertSub = componentRef.instance.close.subscribe(() => 
    {
      this.alertSub?.unsubscribe();
      hostViewContainerRef.clear();
    })

  }
}