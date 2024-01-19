import { Component, ViewChild } from '@angular/core';
import { NgForm } from '@angular/forms';
import { UserPasswordInterface } from 'src/app/shared/models/user-password.interface';
import { PlaceholderDirective } from 'src/app/shared/ui/alert/directive/placeholder.directive';
import { AlertService } from 'src/app/shared/ui/alert/service/alert.service';
import { UserService } from '../user/services/user.service';

@Component({
  selector: 'app-password-reset',
  templateUrl: './password-reset.component.html',
  styleUrls: ['./password-reset.component.scss']
})
export class PasswordResetComponent {

  @ViewChild('form') form : NgForm | null = null;
  @ViewChild(PlaceholderDirective, { static: true }) alertHost!: PlaceholderDirective;

  isLoading : boolean = false;
  userData : UserPasswordInterface = { email : '', new_password : '', re_new_password : '' }

  constructor(private userService : UserService, private alertService : AlertService){}

  Submit() : void
  {
    if(this.form?.valid === false)
      return

      this.isLoading = true;

      this.userService.ResetPassword(this.userData)
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
