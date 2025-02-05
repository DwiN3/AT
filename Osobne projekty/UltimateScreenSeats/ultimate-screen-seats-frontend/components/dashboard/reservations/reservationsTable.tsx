"use client";

import React, { useState, useMemo } from "react";
import { Table, TableBody, TableCell, TableColumn, TableHeader, TableRow } from "@heroui/table";
import { Input } from "@nextui-org/input";
import { Pagination } from "@heroui/pagination";
import { Tooltip } from "@nextui-org/tooltip";
import { Spinner } from "@nextui-org/spinner";

interface ReservationsTableProps {
  reservations: Reservation[];
  loading: boolean;
}

export default function ReservationsTable({ reservations, loading }: ReservationsTableProps) {
  const [filterValue, setFilterValue] = useState<string>("");
  const [page, setPage] = useState<number>(1);
  const rowsPerPage = 5;

  const filteredReservations = useMemo(() => {
    if (!filterValue) return reservations;

    return reservations.filter((reservation) =>
      reservation.showing.movie.title.toLowerCase().includes(filterValue.toLowerCase())
    );
  }, [reservations, filterValue]);

  const paginatedReservations = useMemo(() => {
    const start = (page - 1) * rowsPerPage;
    const end = start + rowsPerPage;

    return filteredReservations.slice(start, end);
  }, [filteredReservations, page, rowsPerPage]);

  const totalPages = Math.ceil(filteredReservations.length / rowsPerPage);

  const renderSeatLayout = (layout: number[][], seatRow: number, seatCol: number) => (
    <div
      className="inline-grid gap-1 p-2"
      style={{ gridTemplateColumns: `repeat(${layout[0].length}, 1fr)` }}
    >
      {layout.flatMap((row, rowIndex) =>
        row.map((seat, colIndex) => {
          if (seat === -1) {
            return (
              <div
                key={`${rowIndex}-${colIndex}`}
                className="w-6 h-6 bg-transparent"
                style={{ visibility: "hidden" }}
              />
            );
          }

          const isReserved = rowIndex === seatRow && colIndex === seatCol;

          return (
            <div
              key={`${rowIndex}-${colIndex}`}
              className={`w-6 h-6 flex items-center justify-center rounded text-xs ${
                isReserved ? "bg-red-500 text-white" : "bg-primary-400"
              }`}
            >
              {isReserved ? "X" : ""}
            </div>
          );
        })
      )}
    </div>
  );

  const renderCell = (reservation: Reservation, columnKey: string) => {
    switch (columnKey) {
      case "id":
        return reservation.id;
      case "customer":
        return (
          <div>
            <p>{reservation.user.username}</p>
            <p>{reservation.user.email}</p>
          </div>
        );
      case "movie":
        return reservation.showing.movie.title;
      case "date":
        return new Date(reservation.showing.date).toLocaleString();
      case "cinema_room":
        return reservation.showing.cinema_room.name;
      case "place":
        return `Rząd ${reservation.seat_row + 1}, Miejsce ${reservation.seat_column + 1}`;
      case "seat_layout":
        return (
          <Tooltip content="Układ sali kinowej">
            {renderSeatLayout(
              reservation.showing.cinema_room.seat_layout,
              reservation.seat_row,
              reservation.seat_column
            )}
          </Tooltip>
        );
      default:
        return null;
    }
  };

  if (loading) {
    return (
      <div className="flex flex-row gap-4 h-full justify-center items-center">
        <Spinner />
        <p className="text-md">Ładowanie Rezerwacji...</p>
      </div>
    );
  }

  return (
    <div className="p-4 w-full">
      <h1 className="text-3xl text-primary font-semibold pb-8">Lista Rezerwacji</h1>
      <div className="flex justify-between items-center mb-4">
        <Input
          className="w-1/3"
          placeholder="Szukaj po nazwie filmu..."
          value={filterValue}
          onChange={(e) => setFilterValue(e.target.value)}
        />
      </div>
      <Table
        isStriped
        bottomContent={
          <div className="flex w-full justify-center">
            <Pagination
              isCompact
              showControls
              showShadow
              color="primary"
              page={page}
              total={totalPages}
              onChange={setPage}
            />
          </div>
        }
        color="primary"
      >
        <TableHeader>
          <TableColumn key="id">ID</TableColumn>
          <TableColumn key="customer">Osoba rezerwująca</TableColumn>
          <TableColumn key="movie">Film</TableColumn>
          <TableColumn key="date">Data</TableColumn>
          <TableColumn key="cinema_room">Sala</TableColumn>
          <TableColumn key="place">Miejsce</TableColumn>
          <TableColumn key="seat_layout">Układ sali</TableColumn>
        </TableHeader>
        <TableBody items={paginatedReservations}>
          {(item) => (
            <TableRow key={item.id}>
              {(columnKey) => (
                <TableCell>
                  {renderCell(item, columnKey as keyof Reservation | "actions")}
                </TableCell>
              )}
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  );
}