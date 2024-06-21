import { Component, OnInit } from '@angular/core';
import { Router } from "@angular/router";
import { FormsModule } from "@angular/forms";
import { CommonModule } from '@angular/common';
import { AuthService } from "../../services/auth/auth.service";

@Component({
  selector: 'app-signup',
  standalone: true,
  imports: [FormsModule, CommonModule],
  providers: [AuthService],
  templateUrl: './signup.component.html',
  styleUrls: ['./signup.component.css']
})
export class SignupComponent implements OnInit {

  public credentials = {
    name: '',
    email: '',
    password: '',
  };

  public repassword: string = '';
  public MessageText: string = '';

  constructor(
    private authService: AuthService, 
    public router: Router) {}

  ngOnInit() {}

  create() {
    this.MessageText = '';
    if(this.credentials.password === this.repassword){
      this.authService.createOrUpdate(this.credentials).subscribe((response) => {
        this.router.navigate(['/login']);
      }, error => {
        if (error.status === 409) {
          this.MessageText = 'Konto o podanym adresie email lub nazwie istnieje';
        } else {
          this.MessageText = 'Wystąpił błąd. Spróbuj ponownie.';
        }
      });
    } else{
      this.MessageText = 'Hasła się różnią';
    }
  }
}