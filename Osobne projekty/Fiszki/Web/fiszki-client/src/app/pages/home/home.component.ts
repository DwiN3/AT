import { Component, ComponentFactoryResolver, OnDestroy, ViewChild } from '@angular/core';
import { NgForm } from '@angular/forms';
import { Title } from '@angular/platform-browser';
import { Router } from '@angular/router';
import { distinctUntilChanged, Subscription } from 'rxjs';
import { BaseUserModel } from 'src/app/shared/models/base-user.model';
import { AccountService } from 'src/app/shared/services/user.service';
import { PlaceholderDirective } from 'src/app/shared/ui/alert/directive/placeholder.directive';
import { AlertService } from 'src/app/shared/ui/alert/service/alert.service';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent implements OnDestroy{

  @ViewChild('form') loginForm : NgForm | null = null;
  @ViewChild(PlaceholderDirective, { static: true }) alertHost!: PlaceholderDirective;

  userData : BaseUserModel = new BaseUserModel('', '');
  error : string | null = null;
  
  subscription : Subscription | null = null;
  isLoading : boolean = false;

  constructor(private accountService : AccountService, private router : Router, private alertService : AlertService, private title : Title)
  {
    this.title.setTitle('Fiszki  |  Logowanie');
  }

  ngAfterViewInit(): void 
  {
    if(this.loginForm)
    {
      this.subscription = this.loginForm.valueChanges ? this.loginForm.valueChanges.pipe(distinctUntilChanged())
        .subscribe(() => {
          this.error = null;
        })
        : null
    }
  }

  ngOnDestroy(): void 
  {
    if(this.subscription)
      this.subscription.unsubscribe();
  }

  Submit() : void
  {
      if(this.loginForm?.valid === false)
        return

      this.isLoading = true;

      this.accountService.Login(this.userData)
        .subscribe(resData => {
            localStorage.setItem('token', JSON.stringify(resData.response).replace(/"/g, ''));
            this.router.navigate(['user']);
          }, error => {
            this.isLoading = false;
            if(error.status === 401)
              this.error = "Zły email lub hasło!";
            else
              this.alertService.ShowAlert('Błąd serwera!', error.message, 'Spróbuj ponownie później!', this.alertHost);
          });
  }

}
