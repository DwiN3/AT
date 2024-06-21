import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { HttpClientModule } from "@angular/common/http";
import { CommonModule } from "@angular/common";
import { Router } from "@angular/router";
import { DataService } from "../../../services/post/post.service";
import { AuthService } from '../../../services/auth/auth.service';

@Component({
  selector: 'blog-item-details',
  standalone: true,
  imports: [HttpClientModule, CommonModule],
  providers: [ DataService ],
  templateUrl: './blog-item-details.component.html',
  styleUrl: './blog-item-details.component.css'
})
export class BlogItemDetailsComponent implements OnInit {
  public image: string = '';
  public text: string = '';
  public title: string = '';
  public id: string = '';

  public isAdmin?: boolean;

  constructor(
    private service: DataService,
    private authService: AuthService, 
    private route: ActivatedRoute,
    private router: Router) {}

  ngOnInit() {
    this.isAdmin = this.authService.isAdmin();

    this.route.paramMap
      .subscribe((params: any) => {
        this.id = params.get('id');
      });

    this.service.getById(this.id).subscribe((res: any) => {
      const post = res;
      this.title = post['title'];
      this.image = post['image'];
      this.text = post['text'];
    });
  }

  removePost() {
    if (this.id != null) {
      this.service.removePost(this.id).subscribe(response => {
      }, error => {});
      this.router.navigate(['/blog']);
    }
  }
}