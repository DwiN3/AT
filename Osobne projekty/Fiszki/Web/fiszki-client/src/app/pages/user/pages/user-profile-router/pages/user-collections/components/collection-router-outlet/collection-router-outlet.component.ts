import { Component } from '@angular/core';
import { ActivatedRoute } from '@angular/router';

@Component({
  selector: 'app-collection-router-outlet',
  templateUrl: './collection-router-outlet.component.html',
  styleUrls: ['./collection-router-outlet.component.scss']
})
export class CollectionRouterOutletComponent {

  constructor(private route: ActivatedRoute) 
  {
    this.route.queryParams
    .subscribe(params => {
      const collectionName = params['collectionName'];
    });
  }

}
