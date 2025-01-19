-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: localhost:3306
-- Generation Time: Jan 19, 2025 at 11:17 AM
-- Server version: 8.0.30
-- PHP Version: 8.1.10

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `coffee_db`
--

-- --------------------------------------------------------

--
-- Table structure for table `login`
--

CREATE TABLE `login` (
  `id` int NOT NULL,
  `username` varchar(255) NOT NULL,
  `password` varchar(255) NOT NULL,
  `role` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `login`
--

INSERT INTO `login` (`id`, `username`, `password`, `role`) VALUES
(1, 'nurariffiya', 'admin123', 'barista');

-- --------------------------------------------------------

--
-- Table structure for table `menu_coffee`
--

CREATE TABLE `menu_coffee` (
  `id` int NOT NULL,
  `tanggal_transaksi` date NOT NULL,
  `customer` varchar(255) NOT NULL,
  `item` varchar(255) NOT NULL,
  `kopi (g)` int NOT NULL,
  `susu (g)` int NOT NULL,
  `evaporasi (g)` int NOT NULL,
  `sauce_caramel (g)` int NOT NULL,
  `cafein (mg)` int NOT NULL,
  `calories` int NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `menu_coffee`
--

INSERT INTO `menu_coffee` (`id`, `tanggal_transaksi`, `customer`, `item`, `kopi (g)`, `susu (g)`, `evaporasi (g)`, `sauce_caramel (g)`, `cafein (mg)`, `calories`) VALUES
(2, '2024-07-09', 'dedi', 'kopi latte', 18, 100, 30, 0, 90, 91),
(3, '2024-07-09', 'arya', 'kopi latte', 18, 100, 30, 0, 90, 91),
(4, '2024-07-09', 'sena', 'vietnam drip', 15, 25, 0, 0, 75, 40),
(5, '2024-07-09', 'arif ', 'hazelnut', 18, 100, 30, 0, 90, 77),
(6, '2024-07-09', 'adit', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(7, '2024-07-09', 'rafli', 'hazelnut', 18, 100, 30, 0, 90, 77),
(8, '2024-07-09', 'faris', 'butterscotch', 18, 100, 30, 0, 90, 92),
(9, '2024-07-09', 'esa', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(10, '2024-07-09', 'faris', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(11, '2024-07-09', 'fadhil', 'butterscotch', 18, 100, 30, 0, 90, 92),
(12, '2024-07-09', 'vina', 'butterscotch', 18, 100, 30, 0, 90, 92),
(13, '2024-08-09', 'rahma', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(14, '2024-08-09', 'anissa', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(15, '2024-08-09', 'anissa', 'butterscotch', 18, 100, 30, 0, 90, 92),
(16, '2024-08-09', 'anissa', 'kopi latte', 18, 100, 30, 0, 90, 91),
(17, '2024-08-09', 'anissa', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(18, '2024-08-09', 'tara', 'americano', 18, 0, 0, 0, 90, 0),
(19, '2024-08-09', 'tara', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(20, '2024-08-09', 'tara', 'butterscotch', 18, 100, 30, 0, 90, 92),
(21, '2024-08-09', 'sabila', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(22, '2024-08-09', 'revan', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(23, '2024-08-09', 'revan', 'hazelnut', 18, 100, 30, 0, 90, 77),
(24, '2024-08-09', 'anna', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(25, '2024-08-09', 'fadya', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(26, '2024-08-09', 'putri', 'americano', 18, 0, 0, 0, 90, 0),
(27, '2024-08-09', 'sabrina', 'hazelnut', 18, 100, 30, 0, 90, 77),
(28, '2024-08-09', 'wulan', 'hazelnut', 18, 100, 30, 0, 90, 77),
(29, '2024-08-09', 'akbar', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(30, '2024-08-09', 'akbar', 'hazelnut', 18, 100, 30, 0, 90, 77),
(31, '2024-08-09', 'jay', 'kopi tubruk', 12, 0, 0, 0, 60, 24),
(32, '2024-08-09', 'random', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(33, '2024-08-09', 'random', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(34, '2024-08-09', 'random', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(35, '2024-08-09', 'ratna', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(36, '2024-10-09', 'azwan', 'hazelnut', 18, 100, 30, 0, 90, 77),
(37, '2024-10-09', 'azwan', 'vietnam drip', 15, 25, 0, 0, 75, 40),
(38, '2024-10-09', 'mulya', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(39, '2024-10-09', 'fadly', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(40, '2024-10-09', 'random', 'americano', 18, 0, 0, 0, 90, 0),
(41, '2024-10-09', 'dwi', 'hazelnut', 18, 100, 30, 0, 90, 77),
(42, '2024-10-09', 'random', 'kopi latte', 18, 100, 30, 0, 90, 91),
(43, '2024-11-09', 'tamara', 'vietnam drip', 15, 25, 0, 0, 75, 40),
(44, '2024-11-09', 'tamara', 'caramel machiato ', 18, 100, 30, 10, 90, 319),
(45, '2024-11-09', 'vira', 'caramel machiato ', 18, 100, 30, 10, 90, 319),
(46, '2024-11-09', 'imam', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(47, '2024-11-09', 'random', 'kopi tubruk', 12, 0, 0, 0, 60, 24),
(48, '2024-11-09', 'ramli', 'americano ', 18, 0, 0, 0, 90, 0),
(49, '2024-11-09', 'random', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(50, '2024-12-09', 'sandi', 'vietnam drip', 15, 25, 0, 0, 75, 40),
(51, '2024-12-09', 'radit', 'kopi tubruk', 12, 0, 0, 0, 60, 24),
(52, '2024-12-09', 'random', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(53, '2024-12-09', 'ipan', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(54, '2024-12-09', 'maldiv', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(55, '2024-12-09', 'vicky', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(56, '2024-12-09', 'bagus', 'kopi tubruk', 12, 0, 0, 0, 60, 24),
(57, '2024-12-09', 'agung', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(58, '2024-12-09', 'agung', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(59, '2024-12-09', 'rama', 'vietnam drip', 15, 25, 0, 0, 75, 40),
(60, '2024-09-13', 'ramli', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(61, '2024-09-13', 'ramzi', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(62, '2024-09-13', 'ramzi', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(63, '2024-09-13', 'vina', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(64, '2024-09-13', 'random', 'hazelnut', 18, 100, 30, 0, 90, 77),
(65, '2024-09-13', 'raka', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(66, '2024-09-13', 'jadzil', 'kopi latte', 18, 100, 30, 0, 90, 91),
(67, '2024-09-14', 'azwan', 'hazelnut ', 18, 100, 30, 0, 90, 77),
(68, '2024-09-14', 'azwan', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(69, '2024-09-14', 'random', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(70, '2024-09-14', 'azwan', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(71, '2024-09-14', 'fajri', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(72, '2024-09-14', 'rama', 'hazelnut', 18, 100, 30, 0, 90, 77),
(73, '2024-09-14', 'rama', 'americano', 18, 0, 0, 0, 90, 0),
(74, '2024-09-14', 'niken', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(75, '2024-09-14', 'huda', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(76, '2024-09-14', 'jacky', 'americano ', 18, 0, 0, 0, 90, 0),
(77, '2024-09-14', 'fikri', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(78, '2024-09-14', 'bu dewi', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(79, '2024-09-14', 'ratu', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(80, '2024-09-14', 'ratu ', 'hazelnut ', 18, 100, 30, 0, 90, 77),
(81, '2024-09-14', 'faiz', 'hazelnut ', 18, 100, 30, 0, 90, 77),
(82, '2024-09-14', 'mahdi', 'Hazelnut ', 18, 100, 30, 0, 90, 77),
(83, '2024-09-14', 'random', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(84, '2024-09-14', 'adam', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(85, '2024-09-14', 'random', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(86, '2024-09-14', 'iky', 'americano ', 18, 0, 0, 0, 90, 0),
(87, '2024-09-14', 'sarah', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(88, '2024-09-14', 'dimas', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(89, '2024-09-14', 'mahdi', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(90, '2024-09-14', 'mahdi', 'hazelnut ', 18, 100, 30, 0, 90, 77),
(91, '2024-09-14', 'nawap ', 'hazelnut', 18, 100, 30, 0, 90, 77),
(92, '2024-09-14', 'bu dewi', 'americano', 18, 0, 0, 0, 90, 0),
(93, '2024-09-15', 'ka zahra', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(94, '2024-09-15', 'mutiara', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(95, '2024-09-15', 'sinta', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(96, '2024-09-15', 'kamila', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(97, '2024-09-15', 'kamila', 'Hazelnut ', 18, 100, 30, 0, 90, 77),
(98, '2024-09-15', 'ka nisa', 'americano ', 18, 0, 0, 0, 90, 0),
(99, '2024-09-15', 'ka nisa', 'Hazelnut ', 18, 100, 30, 0, 90, 77),
(100, '2024-09-15', 'ka nisa', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(101, '2024-09-15', 'Ayu', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(102, '2024-09-15', 'Budi', 'Hazelnut ', 18, 100, 30, 0, 90, 77),
(103, '2024-09-15', 'Citra', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(104, '2024-09-15', 'Dewi', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(105, '2024-09-15', 'Eko', 'Hazelnut ', 18, 100, 30, 0, 90, 77),
(106, '2024-09-15', 'Fajar', 'Hazelnut ', 18, 100, 30, 0, 90, 77),
(107, '2024-09-15', 'Gita', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(108, '2024-09-17', 'Hendra', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(109, '2024-09-17', 'Intan', 'Hazelnut ', 18, 100, 30, 0, 90, 77),
(110, '2024-09-17', 'Joko', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(111, '2024-09-17', 'Kartini', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(112, '2024-09-17', 'Lestari', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(113, '2024-09-17', 'Made', 'americano ', 18, 0, 0, 0, 90, 0),
(114, '2024-09-17', 'Nina', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(115, '2024-09-17', 'Oka', 'americano ', 18, 0, 0, 0, 90, 0),
(116, '2024-09-17', 'Putri', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(117, '2024-09-17', 'Rini', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(118, '2024-09-17', 'Siti', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(119, '2024-09-17', 'Tono', 'Hazelnut ', 18, 100, 30, 0, 90, 77),
(120, '2024-09-18', 'Umar', 'americano ', 18, 0, 0, 0, 90, 0),
(121, '2024-09-18', 'Vina', 'Hazelnut ', 18, 100, 30, 0, 90, 77),
(122, '2024-09-18', 'Wawan', 'americano ', 18, 0, 0, 0, 90, 0),
(123, '2024-09-18', 'Yuli', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(124, '2024-09-18', 'Zaki', 'americano ', 18, 0, 0, 0, 90, 0),
(125, '2024-09-18', 'Agus', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(126, '2024-09-18', 'Bayu', 'vanilla latte', 18, 100, 30, 0, 90, 112),
(127, '2024-09-18', 'Chandra', 'americano ', 18, 0, 0, 0, 90, 0),
(128, '2024-09-18', 'Dani', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(129, '2024-09-18', 'Erna', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(130, '2024-09-18', 'Firman', 'Hazelnut ', 18, 100, 30, 0, 90, 77),
(131, '2024-09-18', 'Gilang', 'kopi tubruk', 12, 0, 0, 0, 60, 24),
(132, '2024-09-18', 'Halim', 'Hazelnut ', 18, 100, 30, 0, 90, 77),
(133, '2024-09-18', 'Ira', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(134, '2024-09-18', 'Jamil', 'vietnam drip', 15, 25, 0, 0, 75, 40),
(135, '2024-09-18', 'Krisna', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(136, '2024-09-18', 'Maya', 'americano ', 18, 0, 0, 0, 90, 0),
(137, '2024-09-19', 'Novi', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(138, '2024-09-19', 'Okta', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(139, '2024-09-19', 'Prima', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(140, '2024-09-19', 'Rama', 'americano ', 18, 0, 0, 0, 90, 0),
(141, '2024-09-19', 'Samsul', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(142, '2024-09-19', 'Tri', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(143, '2024-09-19', 'Utami', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(144, '2024-09-19', 'Vera', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(145, '2024-09-19', 'Wira', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(146, '2024-09-19', 'Yogi', 'americano ', 18, 0, 0, 0, 90, 0),
(147, '2024-09-20', 'Zul', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(148, '2024-09-20', 'Ayu', 'kopi tubruk', 12, 0, 0, 0, 60, 24),
(149, '2024-09-20', 'Budi', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(150, '2024-09-20', 'Citra', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(151, '2024-09-20', 'Dewi', 'hazelnut ', 18, 100, 30, 0, 90, 77),
(152, '2024-09-20', 'Eko', 'hazelnut ', 18, 100, 30, 0, 90, 77),
(153, '2024-09-20', 'Fajar', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(154, '2024-09-20', 'Gita', 'americano ', 18, 0, 0, 0, 90, 0),
(155, '2024-09-20', 'Hendra', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(156, '2024-09-20', 'Intan', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(157, '2024-09-20', 'Joko', 'americano ', 18, 0, 0, 0, 90, 0),
(158, '2024-09-20', 'Kartini', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(159, '2024-09-20', 'Lestari', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(160, '2024-09-20', 'Made', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(161, '2024-09-20', 'Nina', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(162, '2024-09-20', 'Oka', 'hazelnut ', 18, 100, 30, 0, 90, 77),
(163, '2024-09-20', 'Putri', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(164, '2024-09-20', 'Rini', 'caremel machiato', 18, 100, 30, 10, 90, 319),
(165, '2024-09-20', 'Siti', 'americano ', 18, 0, 0, 0, 90, 0),
(166, '2024-09-20', 'Tono', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(167, '2024-09-20', 'Umar', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(168, '2024-09-20', 'Vina', 'hazelnut ', 18, 100, 30, 0, 90, 77),
(169, '2024-09-20', 'Wawan', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(170, '2024-09-20', 'Yuli', 'vietnam drip', 15, 25, 0, 0, 75, 40),
(171, '2024-09-21', 'Zaki', 'hazelnut ', 18, 100, 30, 0, 90, 77),
(172, '2024-09-21', 'Agus', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(173, '2024-09-21', 'Bayu', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(174, '2024-09-21', 'Chandra', 'hazelnut ', 18, 100, 30, 0, 90, 77),
(175, '2024-09-21', 'Dani', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(176, '2024-09-21', 'Erna', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(177, '2024-09-21', 'Firman', 'americano ', 18, 0, 0, 0, 90, 0),
(178, '2024-09-21', 'Gilang', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(179, '2024-09-21', 'Halim', 'hazelnut ', 18, 100, 30, 0, 90, 77),
(180, '2024-09-21', 'Ira', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(181, '2024-09-21', 'Jamil', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(182, '2024-09-21', 'Krisna', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(183, '2024-09-21', 'Luki', 'hazelnut ', 18, 100, 30, 0, 90, 77),
(184, '2024-09-21', 'Maya', 'americano ', 18, 0, 0, 0, 90, 0),
(185, '2024-09-21', 'Novi', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(186, '2024-09-21', 'Okta', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(187, '2024-09-21', 'Prima', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(188, '2024-09-21', 'Rama', 'americano ', 18, 0, 0, 0, 90, 0),
(189, '2024-09-21', 'Samsul', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(190, '2024-09-21', 'Tri', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(191, '2024-09-21', 'Utami', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(192, '2024-09-21', 'Vera', 'kopi tubruk', 12, 0, 0, 0, 60, 24),
(193, '2024-09-21', 'Wira', 'kopi tubruk', 12, 0, 0, 0, 60, 24),
(194, '2024-09-21', 'Yogi', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(195, '2024-09-21', 'Zul', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(196, '2024-09-21', 'Ayu', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(197, '2024-09-21', 'Budi', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(198, '2024-09-21', 'Citra', 'americano ', 18, 0, 0, 0, 90, 0),
(199, '2024-09-21', 'Dewi', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(200, '2024-09-21', 'Eko', 'Hazelnut ', 18, 100, 30, 0, 90, 77),
(201, '2024-09-21', 'Fajar', 'americano ', 18, 0, 0, 0, 90, 0),
(202, '2024-09-21', 'Gita', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(203, '2024-09-21', 'Hendra', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(204, '2024-09-22', 'Intan', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(205, '2024-09-22', 'Joko', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(206, '2024-09-22', 'Kartini', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(207, '2024-09-22', 'Lestari', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(208, '2024-09-22', 'Made', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(209, '2024-09-22', 'Nina', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(210, '2024-09-22', 'Oka', 'Hazelnut ', 18, 100, 30, 0, 90, 77),
(211, '2024-09-22', 'Putri', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(212, '2024-09-22', 'Rini', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(213, '2024-09-22', 'Siti', 'Hazelnut ', 18, 100, 30, 0, 90, 77),
(214, '2024-09-22', 'Tono', 'Hazelnut ', 18, 100, 30, 0, 90, 77),
(215, '2024-09-22', 'Umar', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(216, '2024-09-22', 'Vina', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(217, '2024-09-22', 'Wawan', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(218, '2024-09-22', 'Yuli', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(219, '2024-09-22', 'Zaki', 'Hazelnut ', 18, 100, 30, 0, 90, 77),
(220, '2024-09-22', 'Agus', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(221, '2024-09-22', 'Bayu', 'Hazelnut ', 18, 100, 30, 0, 90, 77),
(222, '2024-09-22', 'Chandra', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(223, '2024-09-22', 'Dani', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(224, '2024-09-22', 'Erna', 'americano ', 18, 0, 0, 0, 90, 0),
(225, '2024-09-22', 'Firman', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(226, '2024-09-22', 'Gilang', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(227, '2024-09-22', 'Halim', 'kopi tubruk', 12, 0, 0, 0, 60, 24),
(228, '2024-09-24', 'Ira', 'hazelnut ', 18, 100, 30, 0, 90, 77),
(229, '2024-09-24', 'Jamil', 'americano ', 18, 0, 0, 0, 90, 0),
(230, '2024-09-24', 'Krisna', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(231, '2024-09-24', 'Luki', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(232, '2024-09-24', 'Maya', 'kopi tubruk', 12, 0, 0, 0, 60, 24),
(233, '2024-09-24', 'Novi', 'americano ', 18, 0, 0, 0, 90, 0),
(234, '2024-09-25', 'Okta', 'americano ', 18, 0, 0, 0, 90, 0),
(235, '2024-09-25', 'Prima', 'vanilla latte', 18, 100, 30, 0, 90, 112),
(236, '2024-09-25', 'Rama', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(237, '2024-09-25', 'Samsul', 'hazelnut ', 18, 100, 30, 0, 90, 77),
(238, '2024-09-25', 'Tri', 'americano ', 18, 0, 0, 0, 90, 0),
(239, '2024-09-25', 'Utami', 'hazelnut ', 18, 100, 30, 0, 90, 77),
(240, '2024-09-25', 'Vera', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(241, '2024-09-25', 'Wira', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(242, '2024-09-25', 'Yogi', 'hazelnut ', 18, 100, 30, 0, 90, 77),
(243, '2024-09-25', 'Zul', 'hazelnut ', 18, 100, 30, 0, 90, 77),
(244, '2024-09-26', 'Ayu', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(245, '2024-09-26', 'Budi', 'kopi tubruk', 12, 0, 0, 0, 60, 24),
(246, '2024-09-26', 'Citra', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(247, '2024-09-26', 'Dewi', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(248, '2024-09-26', 'Eko', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(249, '2024-09-26', 'Fajar', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(250, '2024-09-26', 'Gita', 'americano ', 18, 0, 0, 0, 90, 0),
(251, '2024-09-26', 'Hendra', 'hazelnut ', 18, 100, 30, 0, 90, 77),
(252, '2024-09-26', 'Intan', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(253, '2024-09-26', 'Joko', 'hazelnut ', 18, 100, 30, 0, 90, 77),
(254, '2024-09-27', 'Kartini', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(255, '2024-09-27', 'Lestari', 'americano ', 18, 0, 0, 0, 90, 0),
(256, '2024-09-27', 'Made', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(257, '2024-09-27', 'Nina', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(258, '2024-09-27', 'Oka', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(259, '2024-09-27', 'Putri', 'hazelnut ', 18, 100, 30, 0, 90, 77),
(260, '2024-09-27', 'Rini', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(261, '2024-09-27', 'Siti', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(262, '2024-09-27', 'Tono', 'hazelnut ', 18, 100, 30, 0, 90, 77),
(263, '2024-09-27', 'Umar', 'hazelnut ', 18, 100, 30, 0, 90, 77),
(264, '2024-09-27', 'Vina', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(265, '2024-09-27', 'Wawan', 'americano ', 18, 0, 0, 0, 90, 0),
(266, '2024-09-27', 'Yuli', 'americano ', 18, 0, 0, 0, 90, 0),
(267, '2024-09-27', 'Zaki', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(268, '2024-09-28', 'Agus', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(269, '2024-09-28', 'Bayu', 'vanilla latte', 18, 100, 30, 0, 90, 112),
(270, '2024-09-28', 'Chandra', 'kopi tubruk', 12, 0, 0, 0, 60, 24),
(271, '2024-09-28', 'Dani', 'kopi latte ', 18, 100, 30, 0, 90, 91),
(272, '2024-09-28', 'Erna', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(273, '2024-09-28', 'Firman', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(274, '2024-09-28', 'Gilang', 'kopi latte', 18, 100, 30, 0, 90, 91),
(275, '2024-09-28', 'Halim', 'americano ', 18, 0, 0, 0, 90, 0),
(276, '2024-09-28', 'Ira', 'americano ', 18, 0, 0, 0, 90, 0),
(277, '2024-09-28', 'Jamil', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(278, '2024-09-28', 'Krisna', 'kopi latte', 18, 100, 30, 0, 90, 91),
(279, '2024-09-28', 'Luki', 'kopi latte', 18, 100, 30, 0, 90, 91),
(280, '2024-09-28', 'Maya', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(281, '2024-09-28', 'Novi', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(282, '2024-09-28', 'Okta', 'vanilla latte', 18, 100, 30, 0, 90, 112),
(283, '2024-09-28', 'Prima', 'Hazelnut ', 18, 100, 30, 0, 90, 77),
(284, '2024-09-28', 'Rama', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(285, '2024-09-28', 'Samsul', 'kopi latte', 18, 100, 30, 0, 90, 91),
(286, '2024-09-28', 'Tri', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(287, '2024-09-28', 'Utami', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(288, '2024-09-29', 'Vera', 'kopi latte', 18, 100, 30, 0, 90, 91),
(289, '2024-09-29', 'Wira', 'Hazelnut ', 18, 100, 30, 0, 90, 77),
(290, '2024-09-29', 'Yogi', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(291, '2024-09-29', 'Zul', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(292, '2024-09-29', 'Ayu', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(293, '2024-09-29', 'Budi', 'vanilla latte', 18, 100, 30, 0, 90, 112),
(294, '2024-09-29', 'Citra', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(295, '2024-09-29', 'Dewi', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(296, '2024-09-29', 'Eko', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(297, '2024-09-29', 'Fajar', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(298, '2024-09-29', 'Gita', 'Hazelnut ', 18, 100, 30, 0, 90, 77),
(299, '2024-09-29', 'Hendra', 'kopi tubruk', 12, 0, 0, 0, 60, 24),
(300, '2024-09-29', 'Intan', 'butterscotch ', 18, 100, 30, 0, 90, 92),
(301, '2024-09-29', 'Joko', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(302, '2024-09-29', 'Kartini', 'americano ', 18, 0, 0, 0, 90, 0),
(303, '2024-01-10', 'Lestari', 'vanilla latte', 18, 100, 30, 0, 90, 112),
(304, '2024-01-10', 'Made', 'kopi tubruk', 12, 0, 0, 0, 60, 24),
(305, '2024-01-10', 'Nina', 'kopi latte', 18, 100, 30, 0, 90, 91),
(306, '2024-01-10', 'Oka', 'Hazelnut ', 18, 100, 30, 0, 90, 77),
(307, '2024-01-10', 'Putri', 'kopi tubruk', 12, 0, 0, 0, 60, 24),
(308, '2024-01-10', 'Rini', 'Hazelnut ', 18, 100, 30, 0, 90, 77),
(309, '2024-01-10', 'Siti', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(310, '2024-01-10', 'Tono', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(311, '2024-01-10', 'Umar', 'Hazelnut ', 18, 100, 30, 0, 90, 77),
(312, '2024-01-10', 'Vina', 'americano ', 18, 0, 0, 0, 90, 0),
(313, '2024-01-10', 'Wawan', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(314, '2024-01-10', 'Yuli', 'kopi latte', 18, 100, 30, 0, 90, 91),
(315, '2024-01-10', 'Zaki', 'caramel machiato', 18, 100, 30, 10, 90, 319),
(316, '2024-01-10', 'Agus', 'kopi gula aren', 18, 100, 30, 0, 90, 69),
(317, '2024-01-10', 'Bayu', 'kopi latte', 18, 100, 30, 0, 90, 91),
(318, '2024-01-10', 'Chandra', 'Hazelnut ', 18, 100, 30, 0, 90, 77);

-- --------------------------------------------------------

--
-- Table structure for table `menu_kalori`
--

CREATE TABLE `menu_kalori` (
  `id` int NOT NULL,
  `menu` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `kopi (g)` int NOT NULL,
  `susu (g)` int NOT NULL,
  `evaporasi (g)` int NOT NULL,
  `sauce_caramel (g)` int NOT NULL,
  `cafein (mg)` int NOT NULL,
  `calories` int NOT NULL,
  `kategori_kalori` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `menu_kalori`
--

INSERT INTO `menu_kalori` (`id`, `menu`, `kopi (g)`, `susu (g)`, `evaporasi (g)`, `sauce_caramel (g)`, `cafein (mg)`, `calories`, `kategori_kalori`) VALUES
(1, 'caramel macchiato', 18, 100, 30, 10, 90, 319, 'Tinggi'),
(2, 'americano', 18, 0, 0, 0, 90, 0, 'Rendah'),
(3, 'butterscotch', 18, 100, 30, 0, 90, 92, 'Rendah'),
(4, 'hazelnut', 18, 100, 30, 0, 90, 77, 'Rendah'),
(5, 'coffee latte', 18, 100, 30, 0, 90, 91, 'Rendah'),
(6, 'gula aren ', 18, 100, 30, 0, 90, 69, 'Rendah'),
(7, 'vietnam drip', 15, 25, 0, 0, 75, 40, 'Rendah'),
(8, 'kopi tubruk', 12, 0, 0, 0, 60, 24, 'Rendah'),
(9, 'vanilla latte', 18, 100, 30, 0, 90, 112, 'Tinggi');

-- --------------------------------------------------------

--
-- Table structure for table `pick_up_orders`
--

CREATE TABLE `pick_up_orders` (
  `id` int NOT NULL,
  `customer_name` varchar(255) NOT NULL,
  `coffee_type` varchar(255) NOT NULL,
  `size` varchar(255) NOT NULL,
  `sugar_level` varchar(255) NOT NULL,
  `ice_level` varchar(255) NOT NULL,
  `quantity` int NOT NULL,
  `notes` text,
  `order_date` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `metode_payment` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `pick_up_orders`
--

INSERT INTO `pick_up_orders` (`id`, `customer_name`, `coffee_type`, `size`, `sugar_level`, `ice_level`, `quantity`, `notes`, `order_date`, `metode_payment`) VALUES
(7, 'nurariffiya', 'hazelnut', 'Medium', 'Less Sugar', 'Less Ice', 1, '', '2025-01-18 06:27:22', 'Tunai'),
(8, 'ica', 'americano', 'Medium', 'Less Sugar', 'Normal', 1, '', '2025-01-19 03:36:01', 'Tunai');

-- --------------------------------------------------------

--
-- Table structure for table `rekomendasi`
--

CREATE TABLE `rekomendasi` (
  `id` int NOT NULL,
  `nama` varchar(255) NOT NULL,
  `kalori` int NOT NULL,
  `kategori_kalori` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Dumping data for table `rekomendasi`
--

INSERT INTO `rekomendasi` (`id`, `nama`, `kalori`, `kategori_kalori`) VALUES
(1, 'caramel macchiato', 319, 'kalori tinggi'),
(2, 'americano', 0, 'kalori rendah'),
(3, 'butterscotch', 92, 'kalori tinggi'),
(4, 'hazelnut', 77, 'kalori tinggi'),
(5, 'coffee latte', 91, 'kalori tinggi'),
(6, 'gula aren ', 69, 'kalori tinggi'),
(7, 'vietnam drip', 40, 'kalori rendah'),
(8, 'kopi tubruk', 24, 'kalori rendah'),
(9, 'vanilla latte', 112, 'kalori tinggi');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `login`
--
ALTER TABLE `login`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `menu_coffee`
--
ALTER TABLE `menu_coffee`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `menu_kalori`
--
ALTER TABLE `menu_kalori`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `pick_up_orders`
--
ALTER TABLE `pick_up_orders`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `rekomendasi`
--
ALTER TABLE `rekomendasi`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `login`
--
ALTER TABLE `login`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT for table `menu_coffee`
--
ALTER TABLE `menu_coffee`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=320;

--
-- AUTO_INCREMENT for table `menu_kalori`
--
ALTER TABLE `menu_kalori`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=10;

--
-- AUTO_INCREMENT for table `pick_up_orders`
--
ALTER TABLE `pick_up_orders`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=9;

--
-- AUTO_INCREMENT for table `rekomendasi`
--
ALTER TABLE `rekomendasi`
  MODIFY `id` int NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=10;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
