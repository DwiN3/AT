import { Component, ComponentFactoryResolver, OnDestroy, ViewChild, ViewContainerRef } from '@angular/core';
import { NgForm } from '@angular/forms';
import { distinctUntilChanged, Subscription } from 'rxjs';
import { AlertModel } from 'src/app/shared/models/alert.model';
import { RegisterUserModel } from 'src/app/shared/models/register-user.model';
import { AccountService } from 'src/app/shared/services/user.service';
import { AlertComponent } from 'src/app/shared/ui/alert/alert.component';
import { PlaceholderDirective } from 'src/app/shared/ui/alert/directive/placeholder.directive';

@Component({
  selector: 'app-register',
  templateUrl: './register.component.html',
  styleUrls: ['./register.component.scss']
})
export class RegisterComponent implements OnDestroy{

  @ViewChild('form') registerForm : NgForm | null = null;
  @ViewChild(PlaceholderDirective, { static: true }) alertHost!: PlaceholderDirective;
  error : string | null = null;
  userData : RegisterUserModel = new RegisterUserModel('', '', '', '', '')
  errorSub : Subscription | null = null;
  alertSub : Subscription | null = null;
  alertData : AlertModel = new AlertModel('', '', '');

  constructor(private accountService : AccountService, private componentFactoryResolver : ComponentFactoryResolver){}

  ngAfterViewInit(): void 
  {
    if(this.registerForm)
    {
      this.errorSub = this.registerForm.valueChanges ? this.registerForm.valueChanges
      .pipe(distinctUntilChanged())
        .subscribe(() => {
          this.error = null;
        })
        : null
    }
  }

  ngOnDestroy(): void {
    if(this.errorSub)
      this.errorSub.unsubscribe();
    if(this.alertSub)
      this.alertSub.unsubscribe();
  }

  SubmitForm() : void
  {
    if(this.registerForm?.valid === false)
      return
    
    this.accountService.Register(this.userData)
      .subscribe(resData => {
        this.alertData.title = "Pomyslnie założono konto!"
        this.ShowAlert();
      }, err => {
        this.error = err.error.response;
      })
  }

  private ShowAlert(): void
  {
    const alertCmpFactory = this.componentFactoryResolver.resolveComponentFactory(AlertComponent);
    
    const hostViewContainerRef = this.alertHost?.viewContainerRef;
    hostViewContainerRef?.clear();

    const componentRef = hostViewContainerRef?.createComponent(alertCmpFactory);

    componentRef.instance.title = this.alertData.title;
    componentRef.instance.instructions = this.alertData.instructions;
    componentRef.instance.details = this.alertData.details;

    this.alertSub = componentRef.instance.close.subscribe(() => 
    {
      this.alertSub?.unsubscribe();
      hostViewContainerRef.clear();
    })
  }

}
