"use client";

import React, { useMemo, useState } from "react";
import { Table, TableBody, TableCell, TableColumn, TableHeader, TableRow } from "@heroui/table";
import { Input } from "@nextui-org/input";
import { Button } from "@nextui-org/button";
import { Tooltip } from "@nextui-org/tooltip";
import { Trash2, ArrowDown01, ArrowDown10 } from "lucide-react";
import { Pagination } from "@heroui/pagination";
import { Spinner } from "@nextui-org/spinner";

interface ShowingsTableProps {
  showings: Showing[];
  reservations: Reservation[];
  loading: boolean;
  onCreate: (showing?: Showing) => void;
  onDelete: (showing: Showing) => void;
}

export default function ShowingsTable({
  showings,
  reservations,
  loading,
  onCreate,
  onDelete,
}: ShowingsTableProps) {
  const [filterValue, setFilterValue] = useState<string>("");
  const [page, setPage] = useState<number>(1);
  const [sortKey, setSortKey] = useState<keyof Showing | null>("date");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("asc");
  const rowsPerPage = 5;

  const filteredShowings = useMemo(() => {
    if (!filterValue) return showings;

    return showings.filter((showing) =>
      showing.movie.title.toLowerCase().includes(filterValue.toLowerCase())
    );
  }, [showings, filterValue]);

  const sortShowings = (data: Showing[]) => {
    if (!sortKey) return data;

    return [...data].sort((a, b) => {
      const valA = a[sortKey];
      const valB = b[sortKey];

      if (typeof valA === "string" && typeof valB === "string") {
        return sortOrder === "asc" ? valA.localeCompare(valB) : valB.localeCompare(valA);
      }

      if (typeof valA === "number" && typeof valB === "number") {
        return sortOrder === "asc" ? valA - valB : valB - valA;
      }

      return 0;
    });
  };

  const handleSort = (key: keyof Showing) => {
    if (sortKey === key) {
      setSortOrder((prevOrder) => (prevOrder === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortOrder("asc");
    }
  };

  const sortedShowings = useMemo(() => sortShowings(filteredShowings), [filteredShowings, sortKey, sortOrder]);

  const paginatedShowings = useMemo(() => {
    const start = (page - 1) * rowsPerPage;
    const end = start + rowsPerPage;

    return sortedShowings.slice(start, end);
  }, [sortedShowings, page, rowsPerPage]);

  const totalPages = Math.ceil(filteredShowings.length / rowsPerPage);

  const getReservationsForShowing = (showingId: number) => {
    return reservations.filter((res) => res.showing.id === showingId);
  };

  const renderSeatLayout = (
    layout: number[][],
    reservationsForShowing: { seat_row: number; seat_column: number }[]
  ) => (
    <div
      className="inline-grid gap-1"
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
  
          const isReserved = reservationsForShowing.some(
            (res) => res.seat_row === rowIndex && res.seat_column === colIndex
          );
  
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
  
  const renderCell = (showing: Showing, columnKey: keyof Showing | "actions" | "seat_layout") => {
    switch (columnKey) {
      case "id":
        return showing.id;
      case "movie":
        return showing.movie.title;
      case "ticket_price":
        return showing.ticket_price +" zł";
      case "cinema_room":
        return showing.cinema_room.name;
      case "date":
        return new Date(showing.date).toLocaleString();
      case "seat_layout": {
        const reservationsForShowing = getReservationsForShowing(showing.id).map((res) => ({
          seat_row: res.seat_row,
          seat_column: res.seat_column,
        }));

        return renderSeatLayout(showing.cinema_room.seat_layout, reservationsForShowing);
      }
      case "actions":
        return (
          <div className="flex items-center gap-4">
            <Tooltip content="Usuń seans">
              <Trash2
                className="cursor-pointer text-danger"
                onClick={() => onDelete(showing)}
              />
            </Tooltip>
          </div>
        );
      default:
        return null;
    }
  };

  if (loading) {
    return (
      <div className="flex flex-row gap-4 h-full justify-center items-center">
        <Spinner />
        <p className="text-md">Ładowanie Seansów...</p>
      </div>
    );
  }

  return (
    <div className="p-4 w-full">
      <h1 className="text-3xl text-primary font-semibold pb-8">Zarządzanie Seansami</h1>
      <div className="flex justify-between items-center mb-4">
        <Input
          className="w-1/3"
          placeholder="Szukaj po tytule filmu..."
          value={filterValue}
          onChange={(e) => setFilterValue(e.target.value)}
        />
        <Button size="md" onClick={() => onCreate()}>
          Dodaj Nowy Seans
        </Button>
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
        selectionMode="single"
      >
        <TableHeader>
          <TableColumn key="id" onClick={() => handleSort("id")}>
            <div className="flex items-center gap-1 cursor-pointer">
              ID {sortKey === "id" && (sortOrder === "asc" ? <ArrowDown01 /> : <ArrowDown10 />)}
            </div>
          </TableColumn>
          <TableColumn key="movie">Film</TableColumn>
          <TableColumn key="ticket_price" onClick={() => handleSort("ticket_price")}>
            <div className="flex items-center gap-1 cursor-pointer">
              Koszt biletu {sortKey === "ticket_price" && (sortOrder === "asc" ? <ArrowDown01 /> : <ArrowDown10 />)}
            </div>
          </TableColumn>
          <TableColumn key="cinema_room">Sala</TableColumn>
          <TableColumn key="date" onClick={() => handleSort("date")}>
            <div className="flex items-center gap-1 cursor-pointer">
              Data {sortKey === "date" && (sortOrder === "asc" ? <ArrowDown01 /> : <ArrowDown10 />)}
            </div>
          </TableColumn>
          <TableColumn key="seat_layout">Układ sali</TableColumn>
          <TableColumn key="actions">Akcje</TableColumn>
        </TableHeader>
        <TableBody items={paginatedShowings}>
          {(item) => (
            <TableRow key={item.id}>
              {(columnKey) => (
                <TableCell>{renderCell(item, columnKey as keyof Showing | "actions" | "seat_layout")}</TableCell>
              )}
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  );
}
