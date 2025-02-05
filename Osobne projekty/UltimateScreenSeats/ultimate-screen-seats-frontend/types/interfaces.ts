// eslint-disable-next-line @typescript-eslint/no-unused-vars
interface Movie {
    id: number
    title: string
    description: string
    genre: Genre[]
    movie_length: number
    age_classification: number
    image: string
    release_date: string
    background_image: string
    trailer_url: string
    cast: string
    director: string
}

interface Genre {
    id: number
    name: string
}

interface MoviePreview {
    id: number
    title: string
    image: string
    movie_length: number
    age_classification: number
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
interface ShowingList {
    id: number
    date_from: string
    date_to: string
    movie: MoviePreview
}

interface CinemaRoom {
    id: number
    name: string
    seat_layout: number[][]
    number_of_seats: number
}

interface Showing {
    id: number
    movie: Movie
    cinema_room: CinemaRoom
    date: string
    ticket_price: number
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
interface Reservation {
    id: number
    user: User
    showing:  Showing
    seat_row: number
    seat_column: number
    reserve_at: string
}

interface User {
    id: number
    username: string
    email: string
    role: string
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
interface ReservationSeat {
    showing_id: number
    seat_row: number
    seat_column: number
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
interface GroupedSeats {
    showing: Showing
    seats: ReservationSeat[]
}