"use client";

import React, { useState, useMemo } from "react";
import { Table, TableBody, TableCell, TableColumn, TableHeader, TableRow } from "@heroui/table";
import { Button } from "@nextui-org/button";
import { Tooltip } from "@nextui-org/tooltip";
import { Edit3, Trash2, ArrowDownAz, ArrowDownZa, ArrowDown01, ArrowDown10 } from "lucide-react";
import { Pagination } from "@heroui/pagination";
import { Spinner } from "@nextui-org/spinner";
import { Input } from "@nextui-org/input";


interface HallTableProps {
  halls: CinemaRoom[];
  loading: boolean;
  onUpdate: (hall?: CinemaRoom) => void;
  onDelete: (hall: CinemaRoom) => void;
}

export default function HallTable({ halls, loading, onUpdate, onDelete }: HallTableProps) {
  const [filterValue, setFilterValue] = useState<string>("");
  const [page, setPage] = useState<number>(1);
  const [sortKey, setSortKey] = useState<keyof CinemaRoom | null>("name");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("asc");
  const rowsPerPage = 3;

  const filteredHalls = useMemo(() => {
    if (!filterValue) return halls;

    return halls.filter((hall) =>
      hall.name.toLowerCase().includes(filterValue.toLowerCase())
    );
  }, [halls, filterValue]);

  const sortHalls = (data: CinemaRoom[]) => {
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

  const handleSort = (key: keyof CinemaRoom) => {
    if (sortKey === key) {
      setSortOrder((prevOrder) => (prevOrder === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortOrder("asc");
    }
  };

  const sortedHalls = useMemo(() => sortHalls(filteredHalls), [filteredHalls, sortKey, sortOrder]);

  const paginatedHalls = useMemo(() => {
    const start = (page - 1) * rowsPerPage;
    const end = start + rowsPerPage;

    return sortedHalls.slice(start, end);
  }, [sortedHalls, page, rowsPerPage]);

  const totalPages = Math.ceil(filteredHalls.length / rowsPerPage);

  const renderSeatLayout = (layout: number[][]) => (
    <div className="inline-grid gap-1" style={{ gridTemplateColumns: `repeat(${layout[0].length}, 1fr)` }}>
      {layout.flat().map((seat, index) => (
        <div
          key={index}
          className={`w-6 h-6 flex items-center justify-center rounded text-xs ${
            seat === -1
              ? "bg-gray-500"

              : "bg-primary-400"
          }`}
        />
      ))}
    </div>
  );

  const renderCell = (hall: CinemaRoom, columnKey: keyof CinemaRoom | "actions") => {
    switch (columnKey) {
      case "id":
        return hall.id;
      case "name":
        return <strong className="text-primary">{hall.name}</strong>;
      case "number_of_seats":
        return hall.number_of_seats;
      case "seat_layout":
        return <Tooltip content="Układ siedzeń">{renderSeatLayout(hall.seat_layout)}</Tooltip>;
      case "actions":
        return (
          <div className="flex items-center gap-4">
            <Tooltip content="Edit hall">
              <Edit3
                className="cursor-pointer text-success"
                onClick={() => onUpdate(hall)}
              />
            </Tooltip>
            <Tooltip content="Delete hall">
              <Trash2
                className="cursor-pointer text-danger"
                onClick={() => onDelete(hall)}
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
        <p className="text-md">Ładowanie Sal...</p>
      </div>
    );
  }

  return (
    <div className="p-4 w-full">
      <h1 className="text-3xl text-primary font-semibold pb-8">Zarządzanie Salami</h1>
      <div className="flex justify-between items-center mb-4">
        <Input
          className="w-1/3"
          placeholder="Szukaj po nazwie sali..."
          value={filterValue}
          onChange={(e) => setFilterValue(e.target.value)}
        />
        <Button size="md" onClick={() => onUpdate()}>
          Dodaj nową salę kinową
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
          <TableColumn key="name" onClick={() => handleSort("name")}>
            <div className="flex items-center gap-1 cursor-pointer">
              Sala {sortKey === "name" && (sortOrder === "asc" ? <ArrowDownAz /> : <ArrowDownZa />)}
            </div>
          </TableColumn>
          <TableColumn key="number_of_seats">Liczba miejsc</TableColumn>
          <TableColumn key="seat_layout">Układ Siedzeń</TableColumn>
          <TableColumn key="actions">Akcje</TableColumn>
        </TableHeader>
        <TableBody items={paginatedHalls}>
          {(item) => (
            <TableRow key={item.id}>
              {(columnKey) => (
                <TableCell>{renderCell(item, columnKey as keyof CinemaRoom | "actions")}</TableCell>
              )}
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  );
}
