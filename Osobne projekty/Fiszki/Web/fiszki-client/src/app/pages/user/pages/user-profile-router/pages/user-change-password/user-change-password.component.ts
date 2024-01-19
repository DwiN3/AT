import { Component, ElementRef, ViewChild } from '@angular/core';
import { NgForm } from '@angular/forms';
import { UserService } from 'src/app/pages/user/services/user.service';
import { PlaceholderDirective } from 'src/app/shared/ui/alert/directive/placeholder.directive';
import { AlertService } from 'src/app/shared/ui/alert/service/alert.service';
import { UserPasswordExtendInterface } from './models/user-password-extend';

@Component({
  selector: 'app-user-change-password',
  templateUrl: './user-change-password.component.html',
  styleUrls: ['./user-change-password.component.scss']
})
export class UserChangePasswordComponent {

  @ViewChild('form') form : NgForm | null = null;
  @ViewChild(PlaceholderDirective, { static: true }) alertHost!: PlaceholderDirective;

  isLoading : boolean = false;

  userData : UserPasswordExtendInterface = 
  {
    email : '', 
    password : '', 
    new_password : '', 
    re_new_password : ''
  }

  constructor(private userService : UserService, private alertService : AlertService){}

  Submit() : void
  {
    if(this.form?.valid === false)
      return

      this.isLoading = true;

      this.userService.ChangePassword(this.userData)
        .subscribe(data => {
          this.isLoading = false;
          this.alertService.ShowAlert('Sukces!', 'Pomyślnie zmieniono hasło!', '', this.alertHost);
        }, err => {
          this.isLoading = false;
          if(err.status === 400)
            this.alertService.ShowAlert('Zły email!', 'Użytkownik o podanym e-mailu nie istnieje!', 'Popraw email', this.alertHost);

          else
            this.alertService.ShowAlert('Błąd serwera!', err.message, 'Spróbuj ponownie później!', this.alertHost);
        })
  }

}
