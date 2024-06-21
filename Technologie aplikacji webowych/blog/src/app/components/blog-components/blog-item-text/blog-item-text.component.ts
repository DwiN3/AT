import { Component, Input } from '@angular/core';
import { RouterModule } from "@angular/router";
import { SummaryPipe } from "../../../pipes/summary/summary.pipe";

@Component({
  selector: 'blog-item-text',
  standalone: true,
  imports: [SummaryPipe, RouterModule],
  templateUrl: './blog-item-text.component.html',
  styleUrl: './blog-item-text.component.css'
})
export class BlogItemTextComponent {
  @Input() title?: string;
  @Input() text?: string;
  @Input() id?: string;
}
