import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { Inject } from '@angular/core';
import { CommonModule, DOCUMENT } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { AuthService } from '../../services/auth/auth.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [FormsModule, CommonModule],
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent implements OnInit {

  public credentials = {
    login: '',
    password: ''
  };

  public logged?: boolean;
  public logout?: boolean;
  public MessageText: string = '';

  constructor(
    public authService: AuthService, 
    private router: Router, 
    @Inject(DOCUMENT) private document: Document) {}

  ngOnInit(): void {}

  login() {
    this.MessageText = '';
    this.authService.authenticate(this.credentials).subscribe((result) => {
      if (result) {
        const localStorage = this.document.defaultView?.localStorage;
        localStorage?.setItem('username', this.credentials.login);
        this.router.navigate(['/']);
      }
    }, error => {
      if (error.status === 401) {
        this.MessageText = 'Nieprawidłowe hasło.';
      } else if (error.status === 404) {
        this.MessageText = 'Użytkownik nie został znaleziony.';
      } else {
        this.MessageText = 'Wystąpił błąd. Spróbuj ponownie.';
      }
    });
  }
}
