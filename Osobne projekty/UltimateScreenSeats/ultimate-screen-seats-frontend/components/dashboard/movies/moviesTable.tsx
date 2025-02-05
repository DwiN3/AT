"use client";

import React, { useState, useMemo } from "react";
import { Table, TableBody, TableCell, TableColumn, TableHeader, TableRow } from "@heroui/table";
import { Input } from "@nextui-org/input";
import { Button } from "@nextui-org/button";
import { Tooltip } from "@nextui-org/tooltip";
import { Chip } from "@heroui/chip";
import { EyeIcon, Edit3, Trash2, ArrowDownAz, ArrowDown10, ArrowDownZa, ArrowDown01 } from "lucide-react";
import { Pagination } from "@heroui/pagination";
import { Spinner } from "@nextui-org/spinner";


interface MoviesTableProps {
  movies: Movie[];
  setMovies: React.Dispatch<React.SetStateAction<Movie[]>>;
  onShowDetail: (movie: Movie) => void;
  onUpdate: (movie?: Movie) => void;
  onDelete: (movie: Movie) => void;
  loading: boolean
}

export default function MoviesTable({ movies, onShowDetail, onUpdate, onDelete, loading }: MoviesTableProps) {
  const [filterValue, setFilterValue] = useState<string>("");
  const [page, setPage] = useState<number>(1);
  const [sortKey, setSortKey] = useState<keyof Movie | null>("title");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("asc");
  const rowsPerPage = 10;

  const filteredMovies = useMemo(() => {
    if (!filterValue) return movies;

    return movies.filter((movie) =>
      movie.title.toLowerCase().includes(filterValue.toLowerCase())
    );
  }, [movies, filterValue]);

  const sortMovies = (data: Movie[]) => {
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

  const handleSort = (key: keyof Movie) => {
    if (sortKey === key) {
      setSortOrder((prevOrder) => (prevOrder === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortOrder("asc");
    }
  };

  const sortedMovies = useMemo(() => sortMovies(filteredMovies), [filteredMovies, sortKey, sortOrder]);

  const paginatedMovies = useMemo(() => {
    const start = (page - 1) * rowsPerPage;
    const end = start + rowsPerPage;

    return sortedMovies.slice(start, end);
  }, [sortedMovies, page, rowsPerPage]);

  const totalPages = Math.ceil(filteredMovies.length / rowsPerPage);

  const renderCell = (movie: Movie, columnKey: keyof Movie | "actions") => {
    switch (columnKey) {
      case "id":
        return movie.id;
      case "title":
        return <strong className="text-primary">{movie.title}</strong>;
      case "movie_length":
        return `${movie.movie_length} min`;
      case "age_classification":
        return movie.age_classification;
      case "release_date":
        return new Date(movie.release_date).toLocaleDateString();
      case "genre":
        return movie.genre.map((g) => (
          <Chip key={g.id} className="mr-1 cursor-pointer" size="sm">
            {g.name}
          </Chip>
        ));
      case "actions":
        return (
          <div className="flex items-center gap-4">
            <Tooltip content="Movie details">
              <EyeIcon
                className="cursor-pointer text-secondary"
                onClick={() => onShowDetail(movie)}
              />
            </Tooltip>
            <Tooltip content="Edit movie">
              <Edit3
                className="cursor-pointer text-success"
                onClick={() => onUpdate(movie)}
              />
            </Tooltip>
            <Tooltip content="Delete movie">
              <Trash2
                className="cursor-pointer text-danger"
                onClick={() => onDelete(movie)}
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
        <p className="text-md">Ładowanie Filmów...</p>
      </div>
    );
  }

  return (
    <div className="p-4">
      <h1 className="text-3xl text-primary font-semibold pb-8">Zarządzanie filmami</h1>
      <div className="flex justify-between items-center mb-4">
        <Input
          className="w-1/3"
          placeholder="Wyszukaj film po tytule..."
          value={filterValue}
          onChange={(e) => setFilterValue(e.target.value)}
        />
        <Button size="md" onClick={() => onUpdate()}>
          Dodaj nowy film
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
          <TableColumn key="title" onClick={() => handleSort("title")}>
            <div className="flex items-center gap-1 cursor-pointer">
              Tytuł {sortKey === "title" && (sortOrder === "asc" ? <ArrowDownAz /> : <ArrowDownZa />)}
            </div>
          </TableColumn>
          <TableColumn key="genre">Gatunek/i</TableColumn>
          <TableColumn key="movie_length" onClick={() => handleSort("movie_length")}>
            <div className="flex items-center gap-1 cursor-pointer">
              Długość {sortKey === "movie_length" && (sortOrder === "asc" ? <ArrowDown01 /> : <ArrowDown10 />)}
            </div>
          </TableColumn>
          <TableColumn key="age_classification">Sugerowany wiek</TableColumn>
          <TableColumn key="release_date" onClick={() => handleSort("release_date")}>
            <div className="flex items-center gap-1 cursor-pointer">
              Data wydania {sortKey === "release_date" && (sortOrder === "asc" ? <ArrowDown01 /> : <ArrowDown10 />)}
            </div>
          </TableColumn>
          <TableColumn key="actions">Akcje</TableColumn>
        </TableHeader>
        <TableBody items={paginatedMovies}>
          {(item) => (
            <TableRow key={item.id}>
              {(columnKey) => (
                <TableCell>{renderCell(item, columnKey as keyof Movie | "actions")}</TableCell>
              )}
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  );
}
